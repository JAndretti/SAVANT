/* ============================================================================
 *  savant.c -- SAVANT for the CVRP, minimal version.
 *
 *  SAVANT = savings construction + simulated annealing over truncated
 *  neighbourhoods. This file implements exactly the configuration retained in
 *  the paper, and nothing else:
 *
 *    - R = 4 starts. Start 0 is the parallel savings heuristic of Clarke and
 *      Wright; starts 1..3 perturb every saving by a factor in
 *      [1 - 0.03, 1 + 0.03]. Savings are exact up to n = 1500 customers and
 *      restricted to the 32 nearest neighbours beyond.
 *    - Each start is annealed independently; the best result is returned.
 *      Under --sa-steps S every start runs S steps; under --sa-wall T the
 *      starts share T seconds, one after the other.
 *    - One step draws an operator -- relocate, swap, 2-opt / 2-opt*, a sampled
 *      SWAP* (weight 1 each) or `open` (weight 0.05) -- on a customer u drawn
 *      uniformly and a second customer v drawn uniformly among the K = 20
 *      nearest neighbours of u, and accepts it by the Metropolis rule.
 *    - Every 50 steps, a ruin-and-recreate move (the "kick") removes 1 to 20
 *      customers in strings and reinserts each at its cheapest feasible
 *      position next to one of its 20 nearest neighbours; it is accepted or
 *      undone as a single move.
 *    - T0 is calibrated so that about 0.1 % of the worsening moves are
 *      accepted at the start; T decreases geometrically by one decade over the
 *      run (over the steps, or over the elapsed time under --sa-wall).
 *
 *  It gives the same solutions, bit for bit, as SAVANT/cw.c run with its
 *  default options (the full research version, with every variant studied in
 *  the paper behind an option). Build: `make savant` (see the Makefile).
 *
 *  Representation: one doubly linked list per route, closed on a virtual depot
 *  of its own (indices n+1 .. n+R, at the coordinates of the depot); an empty
 *  route is a virtual depot looping on itself.
 * ==========================================================================*/

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#else
static int  omp_get_max_threads(void) { return 1; }
static void omp_set_num_threads(int t){ (void)t; }
#endif

#if defined(__GNUC__) || defined(__clang__)
#define HOT inline __attribute__((always_inline))
#define COLD __attribute__((noinline))
#else
#define HOT inline
#define COLD
#endif

/* ------------------------------------------------- the retained configuration */
#define RESTARTS     4        /* starts per instance                          */
#define CW_ALPHA     0.03     /* savings perturbation of starts 1..R-1        */
#define CW_EXACT_N   1500     /* exact savings up to this n ...               */
#define CW_KNN       32       /* ... savings on the 32 nearest beyond         */
#define K_SA         20       /* candidate lists of the annealing and kick    */
#define W_RELOCATE   1.0      /* operator weights                             */
#define W_SWAP       1.0
#define W_2OPT       1.0
#define W_SWAPSTAR   1.0
#define W_OPEN       0.05
#define CHI0         0.001    /* initial acceptance of worsening moves        */
#define DECADES      1.0      /* T falls from T0 to T0 * 10^-DECADES          */
#define KICK_EVERY   50       /* a kick every 50 steps ...                    */
#define KICK_MAX     20       /* ... removing 1 to 20 customers               */

#define MAXN 65535u           /* i, j packed into 16 bits each in the savings */
#define EPS  1e-9

/* ------------------------------------------------------------------ utils */

static void die(const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "error: "); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap); exit(1);
}

static void *xmalloc(size_t s)
{
    void *p = malloc(s ? s : 1);
    if (!p) die("out of memory (%zu bytes)", s);
    return p;
}

static void *xrealloc(void *p, size_t s)
{
    void *q = realloc(p, s ? s : 1);
    if (!q) die("out of memory (%zu bytes)", s);
    return q;
}

static long opt_long(const char *opt, const char *s)
{
    char *end;
    errno = 0;
    long long v = strtoll(s, &end, 10);
    if (end == s || *end || errno == ERANGE || v < 0 || v > LONG_MAX)
        die("option %s: expected a non-negative integer, got \"%s\"", opt, s);
    return (long)v;
}

static double opt_num(const char *opt, const char *s)
{
    char *end;
    errno = 0;
    double v = strtod(s, &end);
    if (end == s || *end || !isfinite(v))
        die("option %s: expected a number, got \"%s\"", opt, s);
    return v;
}

static unsigned long long opt_u64(const char *opt, const char *s)
{
    char *end;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s || *end || errno == ERANGE || strchr(s, '-'))
        die("option %s: expected a non-negative integer, got \"%s\"", opt, s);
    return v;
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* --------------------------------------------------------------------- PRNG
 * splitmix64 to seed, xoshiro256** for the stream. One 64-bit draw serves two
 * 32-bit integers, and single bits come from a 64-bit reservoir. */

typedef struct { uint64_t s[4]; uint64_t w32, pool; int has32, nbits; } Rng;

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(Rng *r, uint64_t seed)
{
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&x);
    r->w32 = r->pool = 0; r->has32 = r->nbits = 0;
}

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(Rng *r)
{
    uint64_t *s = r->s;
    const uint64_t result = rotl64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

static inline uint32_t rng32(Rng *r)
{
    if (r->has32) { r->has32 = 0; return (uint32_t)(r->w32 >> 32); }
    r->w32 = rng_next(r); r->has32 = 1;
    return (uint32_t)r->w32;
}

static inline int rng_idx(Rng *r, int n)       /* [0, n), Lemire multiply-shift */
{
    return (int)(((uint64_t)rng32(r) * (uint64_t)n) >> 32);
}

static inline int rng_bit(Rng *r)
{
    if (r->nbits == 0) { r->pool = rng_next(r); r->nbits = 64; }
    int b = (int)(r->pool & 1);
    r->pool >>= 1; r->nbits--;
    return b;
}

static inline double rng_unit(Rng *r)          /* [0, 1) */
{
    return (double)(rng_next(r) >> 11) * 0x1.0p-53;
}

/* ----------------------------------------------------------------- instance
 * index 0 = depot, 1..n = customers. */

typedef struct {
    int     n;
    double  cap;
    double *x, *y, *dem;   /* size n+1 */
    char    name[80];
    int     rounded;       /* integer distances (TSPLIB EUC_2D convention) */
} Inst;

static void inst_alloc(Inst *in, int n)
{
    in->n = n;
    in->x   = (double *)xmalloc((size_t)(n + 1) * sizeof(double));
    in->y   = (double *)xmalloc((size_t)(n + 1) * sizeof(double));
    in->dem = (double *)xmalloc((size_t)(n + 1) * sizeof(double));
}

static void inst_free(Inst *in) { free(in->x); free(in->y); free(in->dem); }

static inline double dist(const Inst *in, int a, int b)
{
    double dx = in->x[a] - in->x[b], dy = in->y[a] - in->y[b];
    double d = sqrt(dx * dx + dy * dy);
    return in->rounded ? floor(d + 0.5) : d;
}

/* CVRPLIB .vrp reader: EUC_2D only, depot from DEPOT_SECTION (node 1 by
   default), integer or real coordinates. Returns 0 on a malformed file. */
static int read_vrp(const char *path, Inst *in, int rounded)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char line[512];
    int dim = -1, depot = 1;
    double cap = -1.0;
    double *X = NULL, *Y = NULL, *D = NULL;

    while (fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (!strncmp(p, "DIMENSION", 9)) {
            char *c = strchr(p, ':'); dim = atoi(c ? c + 1 : p + 9);
            X = (double *)xmalloc((size_t)(dim + 1) * sizeof(double));
            Y = (double *)xmalloc((size_t)(dim + 1) * sizeof(double));
            D = (double *)xmalloc((size_t)(dim + 1) * sizeof(double));
            for (int i = 0; i <= dim; i++) D[i] = 0.0;
        } else if (!strncmp(p, "CAPACITY", 8)) {
            char *c = strchr(p, ':'); cap = atof(c ? c + 1 : p + 8);
        } else if (!strncmp(p, "EDGE_WEIGHT_TYPE", 16)) {
            char *c = strchr(p, ':'); c = c ? c + 1 : p + 16;
            while (*c == ' ' || *c == '\t') c++;
            char t[64]; size_t j = 0;
            while (j + 1 < sizeof t && (isalnum((unsigned char)*c) || *c == '_')) t[j++] = *c++;
            t[j] = 0;
            if (j && strcmp(t, "EUC_2D"))
                die("%s: EDGE_WEIGHT_TYPE %s is not supported (EUC_2D only)", path, t);
        } else if (!strncmp(p, "NODE_COORD_SECTION", 18)) {
            if (dim < 0) goto fail;
            for (int k = 0; k < dim; k++) {
                int id; double a, b;
                if (fscanf(f, "%d %lf %lf", &id, &a, &b) != 3) goto fail;
                if (id < 1 || id > dim) goto fail;
                X[id] = a; Y[id] = b;
            }
        } else if (!strncmp(p, "DEMAND_SECTION", 14)) {
            for (int k = 0; k < dim; k++) {
                int id; double d;
                if (fscanf(f, "%d %lf", &id, &d) != 2) goto fail;
                if (id >= 1 && id <= dim) D[id] = d;
            }
        } else if (!strncmp(p, "DEPOT_SECTION", 13)) {
            int id;
            if (fscanf(f, "%d", &id) == 1 && id >= 1) depot = id;
        }
    }
    fclose(f);
    if (dim <= 1 || cap < 0 || !X) { free(X); free(Y); free(D); return 0; }
    if ((unsigned)(dim - 1) > MAXN) die("%s : n=%d > %u", path, dim - 1, MAXN);

    inst_alloc(in, dim - 1);
    in->cap = cap; in->rounded = rounded;
    in->x[0] = X[depot]; in->y[0] = Y[depot]; in->dem[0] = 0.0;
    int t = 1;
    for (int i = 1; i <= dim; i++) {
        if (i == depot) continue;
        in->x[t] = X[i]; in->y[t] = Y[i]; in->dem[t] = D[i]; t++;
    }
    free(X); free(Y); free(D);
    const char *base = strrchr(path, '/'); base = base ? base + 1 : path;
    snprintf(in->name, sizeof in->name, "%.70s", base);
    return 1;

fail:
    fclose(f);
    free(X); free(Y); free(D);
    return 0;
}

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static int has_suffix(const char *s, const char *suf)
{
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && !strcmp(s + ls - lf, suf);
}

/* every .vrp file of a directory, in name order */
static Inst *read_dir(const char *dirpath, int rounded, int *count)
{
    DIR *d = opendir(dirpath);
    if (!d) die("opening directory %s: %s", dirpath, strerror(errno));
    char **names = NULL; int nn = 0, cap = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!has_suffix(e->d_name, ".vrp")) continue;
        if (nn == cap) { cap = cap ? 2 * cap : 64; names = (char **)xrealloc(names, (size_t)cap * sizeof(char *)); }
        names[nn++] = strdup(e->d_name);
    }
    closedir(d);
    if (!nn) die("no .vrp file in %s", dirpath);
    qsort(names, (size_t)nn, sizeof(char *), cmp_str);

    Inst *v = (Inst *)xmalloc((size_t)nn * sizeof(Inst));
    int nv = 0;
    char path[4096];
    for (int i = 0; i < nn; i++) {
        snprintf(path, sizeof path, "%s/%s", dirpath, names[i]);
        if (read_vrp(path, &v[nv], rounded)) nv++;
        else fprintf(stderr, "warning: %s skipped (unrecognised format)\n", path);
        free(names[i]);
    }
    free(names);
    if (!nv) die("no readable instance in %s", dirpath);
    *count = nv;
    return v;
}

/* =========================================================================
 *  Per-thread workspace, allocated once and reused across instances.
 * ========================================================================= */

typedef struct { uint32_t key; uint32_t ij; } Sav;   /* ij = (i<<16)|j */

typedef struct {
    int     cap_n, cap_k;
    size_t  cap_sav, cap_nbr;
    double *d0;                        /* d(0, i)                             */
    double *xy;                        /* (x, y) per vertex, virtual depots too */
    int     xy_ok, lb_ok, rounded;     /* per-instance caches                 */
    int    *cell_start, *cell_pts;     /* kNN grid                            */
    double *cell_xy;
    int    *nbr; int knn_k;            /* n * knn_k nearest neighbours        */
    double *hd; int *hid;              /* bounded heap of the kNN search      */
    Sav    *sav, *tmp;                 /* savings and radix-sort buffer       */
    int    *uf, *deg, *adj; double *mload;   /* savings merge                 */
    int    *flat, *rstart, *route, *seen;
    int    *s_nxt, *s_prv, *s_rid;  double *s_load;   /* current solution     */
    int    *b_nxt, *b_prv, *b_rid;  double *b_load;   /* best of the chain    */
    int    *r_nxt, *r_prv, *r_rid;  double *r_load;   /* best of the starts   */
    double *inc;                       /* d(prv u, u) + d(u, nxt u)           */
    double *bnx;                       /* d(u, nxt u)                         */
    double *lb;                        /* two shortest edges possible at u    */
    int    *bufA, *bufB, *bufC, *bufD;
    uint32_t *ktag, kick_gen;          /* customers out during a kick         */
    uint32_t th1, th2, th3, th5;       /* operator thresholds on a uint32     */
    long    acc;                       /* accepted moves                      */
    /* --trace: (elapsed s, best cost) pairs of the instance being solved */
    int     tr_on, tr_n, tr_cap;
    double *tr_buf, tr_t0, tr_res, tr_open, tr_best;
} WS;

static void ws_ensure(WS *w, int n, int K, size_t nsav)
{
    if (n > w->cap_n) {
        int c = n + n / 2;
        if (c > (int)MAXN) c = (int)MAXN;
        size_t z = (size_t)2 * c + 4;             /* customers + virtual depots */
        size_t c1 = (size_t)c + 1, c2 = (size_t)c + 2;
        w->d0   = (double *)xrealloc(w->d0,   c1 * sizeof(double));
        w->xy   = (double *)xrealloc(w->xy,   2 * z * sizeof(double));
        w->cell_start = (int *)xrealloc(w->cell_start, (size_t)(4 * c + 16) * sizeof(int));
        w->cell_pts   = (int *)xrealloc(w->cell_pts, c1 * sizeof(int));
        w->cell_xy    = (double *)xrealloc(w->cell_xy, (2 * c1 + 2) * sizeof(double));
        w->uf   = (int *)xrealloc(w->uf,  c1 * sizeof(int));
        w->deg  = (int *)xrealloc(w->deg, c1 * sizeof(int));
        w->adj  = (int *)xrealloc(w->adj, 2 * c1 * sizeof(int));
        w->mload = (double *)xrealloc(w->mload, c1 * sizeof(double));
        w->flat   = (int *)xrealloc(w->flat,   c2 * sizeof(int));
        w->rstart = (int *)xrealloc(w->rstart, c2 * sizeof(int));
        w->route  = (int *)xrealloc(w->route,  c2 * sizeof(int));
        w->seen   = (int *)xrealloc(w->seen,   c1 * sizeof(int));
        w->s_nxt = (int *)xrealloc(w->s_nxt, z * sizeof(int));
        w->s_prv = (int *)xrealloc(w->s_prv, z * sizeof(int));
        w->s_rid = (int *)xrealloc(w->s_rid, z * sizeof(int));
        w->b_nxt = (int *)xrealloc(w->b_nxt, z * sizeof(int));
        w->b_prv = (int *)xrealloc(w->b_prv, z * sizeof(int));
        w->b_rid = (int *)xrealloc(w->b_rid, z * sizeof(int));
        w->r_nxt = (int *)xrealloc(w->r_nxt, z * sizeof(int));
        w->r_prv = (int *)xrealloc(w->r_prv, z * sizeof(int));
        w->r_rid = (int *)xrealloc(w->r_rid, z * sizeof(int));
        w->s_load = (double *)xrealloc(w->s_load, c2 * sizeof(double));
        w->b_load = (double *)xrealloc(w->b_load, c2 * sizeof(double));
        w->r_load = (double *)xrealloc(w->r_load, c2 * sizeof(double));
        w->inc = (double *)xrealloc(w->inc, c2 * sizeof(double));
        w->bnx = (double *)xrealloc(w->bnx, c2 * sizeof(double));
        w->lb  = (double *)xrealloc(w->lb,  c2 * sizeof(double));
        w->bufA = (int *)xrealloc(w->bufA, c2 * sizeof(int));
        w->bufB = (int *)xrealloc(w->bufB, c2 * sizeof(int));
        w->bufC = (int *)xrealloc(w->bufC, c2 * sizeof(int));
        w->bufD = (int *)xrealloc(w->bufD, c2 * sizeof(int));
        w->ktag = (uint32_t *)xrealloc(w->ktag, c2 * sizeof(uint32_t));
        memset(w->ktag, 0, c2 * sizeof(uint32_t));
        w->cap_n = c;
    }
    size_t need = (size_t)n * (size_t)K;
    if (need > w->cap_nbr) { w->nbr = (int *)xrealloc(w->nbr, need * sizeof(int)); w->cap_nbr = need; }
    if (K > w->cap_k) {
        w->hd  = (double *)xrealloc(w->hd,  (size_t)(K + 1) * sizeof(double));
        w->hid = (int *)   xrealloc(w->hid, (size_t)(K + 1) * sizeof(int));
        w->cap_k = K;
    }
    if (nsav > w->cap_sav) {
        w->sav = (Sav *)xrealloc(w->sav, nsav * sizeof(Sav));
        w->tmp = (Sav *)xrealloc(w->tmp, nsav * sizeof(Sav));
        w->cap_sav = nsav;
    }
}

static void ws_free(WS *w)
{
    free(w->d0); free(w->xy); free(w->cell_start); free(w->cell_pts); free(w->cell_xy);
    free(w->nbr); free(w->hd); free(w->hid); free(w->sav); free(w->tmp);
    free(w->uf); free(w->deg); free(w->adj); free(w->mload);
    free(w->flat); free(w->rstart); free(w->route); free(w->seen);
    free(w->s_nxt); free(w->s_prv); free(w->s_rid); free(w->s_load);
    free(w->b_nxt); free(w->b_prv); free(w->b_rid); free(w->b_load);
    free(w->r_nxt); free(w->r_prv); free(w->r_rid); free(w->r_load);
    free(w->inc); free(w->bnx); free(w->lb);
    free(w->bufA); free(w->bufB); free(w->bufC); free(w->bufD); free(w->ktag);
    free(w->tr_buf);
}

/* --trace: best cost against the time elapsed since the instance started,
   noted where the clock is read anyway (every 1024 steps under --sa-wall, and
   after each start). An improvement less than tr_res seconds after the entry
   it would follow is merged into it, with the later time. */
COLD static void trace_note(WS *w, double t_abs, double cost)
{
    if (!(cost < w->tr_best)) return;
    w->tr_best = cost;
    const double t = t_abs - w->tr_t0;
    if (w->tr_n > 0 && t - w->tr_open < w->tr_res) {
        w->tr_buf[2 * w->tr_n - 2] = t;
        w->tr_buf[2 * w->tr_n - 1] = cost;
        return;
    }
    if (w->tr_n == w->tr_cap) {
        w->tr_cap = w->tr_cap ? 2 * w->tr_cap : 256;
        w->tr_buf = (double *)xrealloc(w->tr_buf, (size_t)(2 * w->tr_cap) * sizeof(double));
    }
    w->tr_buf[2 * w->tr_n] = t;
    w->tr_buf[2 * w->tr_n + 1] = cost;
    w->tr_n++;
    w->tr_open = t;
}

/* =========================================================================
 *  k nearest neighbours: uniform grid scanned in rings, brute force for
 *  n <= 512. Each list is sorted, nearest first.
 * ========================================================================= */

static inline void heap_push(double *hd, int *hid, int *sz, int K, double d, int id)
{
    if (*sz < K) {
        int i = (*sz)++;
        hd[i] = d; hid[i] = id;
        while (i > 0) {                        /* sift up in a max-heap */
            int p = (i - 1) >> 1;
            if (hd[p] >= hd[i]) break;
            double td = hd[p]; hd[p] = hd[i]; hd[i] = td;
            int ti = hid[p]; hid[p] = hid[i]; hid[i] = ti;
            i = p;
        }
    } else if (d < hd[0]) {
        hd[0] = d; hid[0] = id;
        int i = 0;
        for (;;) {                             /* sift down */
            int l = 2 * i + 1, r = l + 1, b = i;
            if (l < K && hd[l] > hd[b]) b = l;
            if (r < K && hd[r] > hd[b]) b = r;
            if (b == i) break;
            double td = hd[b]; hd[b] = hd[i]; hd[i] = td;
            int ti = hid[b]; hid[b] = hid[i]; hid[i] = ti;
            i = b;
        }
    }
}

/* heapsort the bounded max-heap in place, then write it nearest first */
static inline void knn_out(double *hd, int *hid, int sz, int *out, int K)
{
    for (int end = sz - 1; end > 0; end--) {
        double td = hd[0]; hd[0] = hd[end]; hd[end] = td;
        int ti = hid[0]; hid[0] = hid[end]; hid[end] = ti;
        for (int i = 0;;) {
            int l = 2 * i + 1, r = l + 1, b = i;
            if (l < end && hd[l] > hd[b]) b = l;
            if (r < end && hd[r] > hd[b]) b = r;
            if (b == i) break;
            double t2 = hd[b]; hd[b] = hd[i]; hd[i] = t2;
            int i2 = hid[b]; hid[b] = hid[i]; hid[i] = i2;
            i = b;
        }
    }
    for (int t = 0; t < K; t++) out[t] = (t < sz) ? hid[t] : 0;
}

static void knn_build(const Inst *in, WS *w, int K)
{
    const int n = in->n;

    if (n <= 512) {                            /* brute force */
        for (int i = 1; i <= n; i++) {
            int sz = 0;
            for (int j = 1; j <= n; j++) {
                if (j == i) continue;
                double dx = in->x[i] - in->x[j], dy = in->y[i] - in->y[j];
                heap_push(w->hd, w->hid, &sz, K, dx * dx + dy * dy, j);
            }
            knn_out(w->hd, w->hid, sz, w->nbr + (size_t)(i - 1) * K, K);
        }
        return;
    }

    double minx = in->x[1], maxx = minx, miny = in->y[1], maxy = miny;
    for (int i = 2; i <= n; i++) {
        if (in->x[i] < minx) minx = in->x[i];
        if (in->x[i] > maxx) maxx = in->x[i];
        if (in->y[i] < miny) miny = in->y[i];
        if (in->y[i] > maxy) maxy = in->y[i];
    }
    int g = (int)sqrt((double)n / 2.0); if (g < 1) g = 1;
    if (g > 2048) g = 2048;
    const int gx = g, gy = g;
    const double W = (maxx - minx), H = (maxy - miny);
    const double cw = (W > 0 ? W : 1.0) / gx, ch = (H > 0 ? H : 1.0) / gy;
    const int ncell = gx * gy;

    int *start = w->cell_start, *pts = w->cell_pts;
    double *pxy = w->cell_xy;                   /* coordinates in cell order */
    for (int c = 0; c <= ncell; c++) start[c] = 0;

    #define CLAMP(v, hi) ((v) < 0 ? 0 : ((v) >= (hi) ? (hi) - 1 : (v)))
    #define CELLX(i) CLAMP((int)((in->x[i] - minx) / cw), gx)
    #define CELLY(i) CLAMP((int)((in->y[i] - miny) / ch), gy)

    for (int i = 1; i <= n; i++) start[CELLY(i) * gx + CELLX(i) + 1]++;
    for (int c = 0; c < ncell; c++) start[c + 1] += start[c];
    {
        int *fill = w->nbr;                     /* temporarily reused */
        for (int c = 0; c < ncell; c++) fill[c] = start[c];
        for (int i = 1; i <= n; i++) pts[fill[CELLY(i) * gx + CELLX(i)]++] = i;
    }
    for (int t = 0; t < n; t++) { int j = pts[t]; pxy[2 * t] = in->x[j]; pxy[2 * t + 1] = in->y[j]; }

    const double step = (cw < ch) ? cw : ch;
    const int maxr = (gx > gy ? gx : gy);
    double *hd = w->hd; int *hid = w->hid;
    for (int qi = 0; qi < n; qi++) {            /* queries in cell order */
        const int i = pts[qi];
        int sz = 0;
        double worst = 1e300;
        const int cx = CELLX(i), cy = CELLY(i);
        const double xi = in->x[i], yi = in->y[i];
        for (int r = 0; r <= maxr; r++) {
            int x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r;
            if (x0 < 0 && x1 >= gx && y0 < 0 && y1 >= gy && r > 0) break;
            for (int vy = (y0 < 0 ? 0 : y0); vy <= (y1 >= gy ? gy - 1 : y1); vy++) {
                int onEdgeY = (vy == y0 || vy == y1);
                for (int vx = (x0 < 0 ? 0 : x0); vx <= (x1 >= gx ? gx - 1 : x1); vx++) {
                    if (!onEdgeY && vx != x0 && vx != x1) {   /* skip the interior */
                        if (x1 < gx) vx = x1 - 1; else vx = gx - 1;
                        continue;
                    }
                    int c = vy * gx + vx;
                    for (int t = start[c]; t < start[c + 1]; t++) {
                        double dx = xi - pxy[2 * t], dy = yi - pxy[2 * t + 1];
                        double d2 = dx * dx + dy * dy;
                        if (d2 >= worst) continue;
                        int j = pts[t];
                        if (j == i) continue;
                        heap_push(hd, hid, &sz, K, d2, j);
                        if (sz == K) worst = hd[0];
                    }
                }
            }
            if (sz >= K) {                       /* lower bound over the remaining rings */
                double b = (double)r * step;
                if (hd[0] <= b * b) break;
            }
        }
        knn_out(hd, hid, sz, w->nbr + (size_t)(i - 1) * K, K);
    }
    #undef CELLX
    #undef CELLY
    #undef CLAMP
}

/* lists of size K, unless those in memory already are */
static void knn_need(const Inst *in, WS *w, int K)
{
    if (w->knn_k == K) return;
    ws_ensure(w, in->n, K, 1);
    knn_build(in, w, K);
    w->knn_k = K;
}

/* =========================================================================
 *  Construction: parallel savings (Clarke & Wright), merged by decreasing
 *  saving with adjacency lists (degree <= 2) and union-find.
 * ========================================================================= */

typedef struct {
    int     n, R;
    int    *nxt, *prv, *rid;
    double *load;
} Sol;

#define VD(r)   (n + 1 + (r))               /* virtual depot of route r */
#define RE(v)   ((v) > n ? 0 : (v))         /* -> real index for dist()  */

static inline int uf_find(int *uf, int i)
{
    int r = i;
    while (uf[r] != r) r = uf[r];
    while (uf[i] != r) { int t = uf[i]; uf[i] = r; i = t; }
    return r;
}

/* LSD radix sort on the float bit pattern of positive savings: stable */
static void radix_sort(Sav *a, Sav *tmp, size_t m)
{
    if (m < 2) return;
    size_t cnt[4][256];
    memset(cnt, 0, sizeof cnt);
    for (size_t i = 0; i < m; i++) {
        uint32_t k = a[i].key;
        cnt[0][ k        & 0xFF]++;
        cnt[1][(k >>  8) & 0xFF]++;
        cnt[2][(k >> 16) & 0xFF]++;
        cnt[3][(k >> 24) & 0xFF]++;
    }
    Sav *src = a, *dst = tmp;
    for (int p = 0; p < 4; p++) {
        int shift = 8 * p;
        if (cnt[p][(src[0].key >> shift) & 0xFF] == m) continue;  /* useless pass */
        size_t pos[256], s = 0;
        for (int b = 0; b < 256; b++) { pos[b] = s; s += cnt[p][b]; }
        for (size_t i = 0; i < m; i++)
            dst[pos[(src[i].key >> shift) & 0xFF]++] = src[i];
        Sav *t = src; src = dst; dst = t;
    }
    if (src != a) memcpy(a, src, m * sizeof(Sav));
}

static inline double dsol(const Inst *in, int n, int a, int b)
{
    return dist(in, RE(a), RE(b));
}

static double sol_cost(const Inst *in, const Sol *S)
{
    const int n = S->n;
    double c = 0.0;
    for (int r = 0; r < S->R; r++) {
        int vd = VD(r);
        for (int t = vd; ; ) {
            int u = S->nxt[t];
            c += dsol(in, n, t, u);
            if (u == vd) break;
            t = u;
        }
    }
    return c;
}

static int route_get(const Sol *S, int r, int *arr)
{
    const int n = S->n;
    int vd = VD(r), L = 0;
    for (int t = S->nxt[vd]; t != vd; t = S->nxt[t]) arr[L++] = t;
    return L;
}

static void route_set(Sol *S, const Inst *in, int r, const int *arr, int L)
{
    const int n = S->n;
    int vd = VD(r), prev = vd;
    double load = 0.0;
    for (int t = 0; t < L; t++) {
        int c = arr[t];
        S->nxt[prev] = c; S->prv[c] = prev; S->rid[c] = r;
        load += in->dem[c]; prev = c;
    }
    S->nxt[prev] = vd; S->prv[vd] = prev;
    S->rid[vd] = r; S->load[r] = load;
}

/* Start rs into *sol; returns its cost. Start 0 is the plain savings
   heuristic, the others multiply every saving by 1 + CW_ALPHA * U[-1, 1]. */
static double build_cw_start(const Inst *in, WS *w, Sol *sol, int rs,
                             uint64_t seed, int exact, int K, int *pvisited)
{
    const int n = sol->n;
    const double *d0 = w->d0;
    Rng rcw; rng_seed(&rcw, seed * 0xD1342543DE82EF95ULL + (uint64_t)rs * 0x9E3779B97F4A7C15ULL);
    const double alpha = (rs > 0) ? CW_ALPHA : 0.0;
    const double cap = in->cap;
    int *flat = w->flat, *rstart = w->rstart, *rt = w->route, *seen = w->seen;
    int R = 0, nf = 0, visited = 0;
    for (int i = 1; i <= n; i++) seen[i] = 0;

    /* --- savings s_ij = d0i + d0j - dij > 0 --- */
    Sav *S = w->sav;
    size_t m = 0;
    if (exact) {
        for (int i = 1; i <= n; i++) {
            double d0i = d0[i];
            for (int j = i + 1; j <= n; j++) {
                double s = d0i + d0[j] - dist(in, i, j);
                if (alpha > 0.0) s *= 1.0 + alpha * (2.0 * rng_unit(&rcw) - 1.0);
                if (s <= 0.0) continue;
                float sf = (float)s; uint32_t k;
                memcpy(&k, &sf, 4);
                S[m].key = k; S[m].ij = ((uint32_t)i << 16) | (uint32_t)j; m++;
            }
        }
    } else {
        knn_need(in, w, K);
        for (int i = 1; i <= n; i++) {
            const int *nb = w->nbr + (size_t)(i - 1) * K;
            double d0i = d0[i];
            for (int t = 0; t < K; t++) {
                int j = nb[t];
                if (j <= 0 || j == i) continue;
                int a = i, b = j;
                if (a > b) { a = j; b = i; }     /* duplicates possible: harmless */
                double s = d0i + d0[j] - dist(in, i, j);
                if (alpha > 0.0) s *= 1.0 + alpha * (2.0 * rng_unit(&rcw) - 1.0);
                if (s <= 0.0) continue;
                float sf = (float)s; uint32_t k;
                memcpy(&k, &sf, 4);
                S[m].key = k; S[m].ij = ((uint32_t)a << 16) | (uint32_t)b; m++;
            }
        }
    }
    radix_sort(S, w->tmp, m);

    /* --- merging, by decreasing saving --- */
    int *uf = w->uf, *deg = w->deg, *adj = w->adj;
    double *load = w->mload;
    for (int i = 1; i <= n; i++) { uf[i] = i; deg[i] = 0; load[i] = in->dem[i]; }
    for (size_t t = m; t-- > 0; ) {
        uint32_t ij = S[t].ij;
        int i = (int)(ij >> 16), j = (int)(ij & 0xFFFF);
        if (deg[i] >= 2 || deg[j] >= 2) continue;    /* not a route endpoint */
        int ri = uf_find(uf, i), rj = uf_find(uf, j);
        if (ri == rj) continue;                      /* same route already   */
        if (load[ri] + load[rj] > cap + EPS) continue;
        adj[2 * i + deg[i]++] = j;
        adj[2 * j + deg[j]++] = i;
        uf[rj] = ri; load[ri] += load[rj];
    }
    for (int s = 1; s <= n; s++) {                   /* walk each path */
        if (seen[s] || deg[s] >= 2) continue;
        int L = 0, prev = 0, cur = s;
        while (cur) {
            rt[++L] = cur; seen[cur] = 1; visited++;
            int nx = 0;
            for (int e = 0; e < deg[cur]; e++)
                if (adj[2 * cur + e] != prev) { nx = adj[2 * cur + e]; break; }
            prev = cur; cur = nx;
        }
        rstart[R++] = nf;
        for (int t = 1; t <= L; t++) flat[nf++] = rt[t];
    }
    rstart[R] = nf;

    sol->R = R;
    for (int r = 0; r < R; r++)
        route_set(sol, in, r, flat + rstart[r], rstart[r + 1] - rstart[r]);
    *pvisited = visited;
    return sol_cost(in, sol);
}

/* =========================================================================
 *  Simulated annealing
 * ========================================================================= */

/* coordinates per vertex, virtual depots materialised: no depot test in dxy */
static inline double dxy(const WS *w, int a, int b)
{
    const double *p = w->xy + 2 * (size_t)a, *q = w->xy + 2 * (size_t)b;
    double dx = p[0] - q[0], dy = p[1] - q[1];
    double d = sqrt(dx * dx + dy * dy);
    return w->rounded ? floor(d + 0.5) : d;
}

static void xy_build(const Inst *in, WS *w)
{
    const int n = in->n;
    for (int i = 0; i <= n; i++) { w->xy[2 * i] = in->x[i]; w->xy[2 * i + 1] = in->y[i]; }
    for (int i = n + 1; i <= 2 * n + 3; i++) { w->xy[2 * i] = in->x[0]; w->xy[2 * i + 1] = in->y[0]; }
    w->rounded = in->rounded;
}

static inline int sa_accept(double delta, double T, Rng *rng)
{
    if (delta <= 0.0) return 1;
    if (delta > 36.0 * T) return 0;
    return rng_unit(rng) < exp(-delta / T);
}

/* lb[u] = the two shortest edges u could carry (depot and its K nearest):
   inc[u] - lb[u] measures how badly u is placed (used by SWAP*) */
static void lb_build(const Inst *in, WS *w, int K)
{
    const int n = in->n;
    for (int u = 1; u <= n; u++) {
        double a = dist(in, 0, u), b = 1e300;
        const int *nb = w->nbr + (size_t)(u - 1) * K;
        for (int t = 0; t < K; t++) {
            int j = nb[t];
            if (j <= 0 || j == u) continue;
            double d = dist(in, u, j);
            if (d < a) { b = a; a = d; }
            else if (d < b) b = d;
        }
        w->lb[u] = (b < 1e299) ? a + b : 2.0 * a;
    }
}

/* d(x, nxt[x]): read from bnx[] for a customer (bit-identical to dxy) */
static inline double dnx(const WS *w, const Sol *S, int x)
{
    if (x <= S->n) return w->bnx[x];
    return (S->nxt[x] <= S->n) ? w->d0[S->nxt[x]] : 0.0;
}

static inline void inc_upd(WS *w, const Sol *S, int u)
{
    if (u < 1 || u > S->n) return;
    double dn = dxy(w, u, S->nxt[u]);
    w->inc[u] = dxy(w, S->prv[u], u) + dn;
    w->bnx[u] = dn;
}

/* Route loads as fresh sums. The O(1) operators update load[] by += / -=,
   which is exact with integer demands; with fractional ones the ULP drift is
   wiped every LOAD_RESYNC steps and at every (re)build of inc[]. */
#define LOAD_RESYNC (1 << 16)

static void loads_resync(const Inst *in, Sol *S)
{
    const int n = S->n;
    const int *nxt = S->nxt;
    for (int r = 0; r < S->R; r++) {
        int vd = VD(r), c = nxt[vd];
        double sl = 0.0;
        for (; c != vd; c = nxt[c]) sl += in->dem[c];
        S->load[r] = sl;
    }
}

static void inc_build(const Inst *in, WS *w, Sol *S)
{
    const int n = S->n;
    loads_resync(in, S);
    for (int u = 1; u <= n; u++) {
        double dn = dxy(w, u, S->nxt[u]);
        w->inc[u] = dxy(w, S->prv[u], u) + dn;
        w->bnx[u] = dn;
    }
}

/* u uniform among the customers; v uniform among the K nearest of u */
static inline int pick_u(const Sol *S, Rng *rng) { return 1 + rng_idx(rng, S->n); }

static inline int sa_cand(const WS *w, int n, int K, Rng *rng, int u)
{
    if (K <= 0) return 1 + rng_idx(rng, n);
    int v = w->nbr[(size_t)(u - 1) * K + rng_idx(rng, K)];
    return (v >= 1 && v <= n) ? v : 1 + rng_idx(rng, n);
}

/* Every operator returns the cost change it applied (0 if rejected), or with
   probe = 1 the change it would apply, without applying it (calibration). */

/* ------------------------------------------------------------- RELOCATE */
static HOT double mv_relocate(const Inst *in, WS *w, Sol *S, int K,
                              double T, Rng *rng, int probe)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;
    int u = pick_u(S, rng);
    int v = sa_cand(w, n, K, rng, u);
    if (v == u) return 0.0;
    if (rng_bit(rng)) v = prv[v];              /* insert before v */
    if (v == u || v == prv[u]) return 0.0;

    int p = prv[u], sc = nxt[u], q = nxt[v];
    int ru = rid[u], rv = rid[v];
    if (ru != rv && S->load[rv] + in->dem[u] > in->cap + EPS) return 0.0;
    double delta = dxy(w, p, sc) - w->inc[u]
                 + dxy(w, v, u) + dxy(w, u, q) - dnx(w, S, v);
    if (probe) return delta;
    if (!sa_accept(delta, T, rng)) return 0.0;

    w->acc++;
    nxt[p] = sc; prv[sc] = p;
    nxt[v] = u;  prv[u]  = v;
    nxt[u] = q;  prv[q]  = u;
    S->load[ru] -= in->dem[u];
    S->load[rv] += in->dem[u];
    rid[u] = rv;
    inc_upd(w, S, p); inc_upd(w, S, sc); inc_upd(w, S, u);
    inc_upd(w, S, v); inc_upd(w, S, q);
    return delta;
}

/* ----------------------------------------------------------------- SWAP */
static HOT double mv_swap(const Inst *in, WS *w, Sol *S, int K,
                          double T, Rng *rng, int probe)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;
    int u = pick_u(S, rng);
    int v = sa_cand(w, n, K, rng, u);
    if (v == u) return 0.0;

    int ru = rid[u], rv = rid[v];
    if (ru != rv) {
        double du = in->dem[u], dv = in->dem[v];
        if (S->load[ru] - du + dv > in->cap + EPS) return 0.0;
        if (S->load[rv] - dv + du > in->cap + EPS) return 0.0;
    }
    int pu = prv[u], su = nxt[u], pv = prv[v], sv = nxt[v];
    double delta;
    if (su == v) {                                   /* u right before v */
        delta = dxy(w, pu, v) + dxy(w, u, sv) - dnx(w, S, pu) - w->bnx[v];
        if (probe) return delta;
        if (!sa_accept(delta, T, rng)) return 0.0;
        w->acc++;
        nxt[pu] = v; prv[v] = pu;
        nxt[v] = u;  prv[u] = v;
        nxt[u] = sv; prv[sv] = u;
    } else if (sv == u) {                            /* v right before u */
        delta = dxy(w, pv, u) + dxy(w, v, su) - dnx(w, S, pv) - w->bnx[u];
        if (probe) return delta;
        if (!sa_accept(delta, T, rng)) return 0.0;
        w->acc++;
        nxt[pv] = u; prv[u] = pv;
        nxt[u] = v;  prv[v] = u;
        nxt[v] = su; prv[su] = v;
    } else {
        delta = dxy(w, pu, v) + dxy(w, v, su) + dxy(w, pv, u) + dxy(w, u, sv)
              - w->inc[u] - w->inc[v];
        if (probe) return delta;
        if (!sa_accept(delta, T, rng)) return 0.0;
        w->acc++;
        nxt[pu] = v; prv[v] = pu; nxt[v] = su; prv[su] = v;
        nxt[pv] = u; prv[u] = pv; nxt[u] = sv; prv[sv] = u;
    }
    if (ru != rv) {
        S->load[ru] += in->dem[v] - in->dem[u];
        S->load[rv] += in->dem[u] - in->dem[v];
        rid[u] = rv; rid[v] = ru;
    }
    inc_upd(w, S, pu); inc_upd(w, S, su); inc_upd(w, S, u);
    inc_upd(w, S, pv); inc_upd(w, S, sv); inc_upd(w, S, v);
    return delta;
}

/* --------------------------------------------------------------- SWAP*
 * Vidal (C&OR 2022): u and v of different routes are exchanged, each going to
 * its best position in the other's route. Here v is not a neighbour of u: the
 * neighbour w0 of u designates the target route, and v is the worse placed
 * (largest inc - lb) of two customers drawn uniformly in it. O(L1 + L2). */
static HOT double mv_swapstar(const Inst *in, WS *w, Sol *S, int K,
                              double T, Rng *rng, int probe)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;
    int u = pick_u(S, rng);
    int w0 = sa_cand(w, n, K, rng, u);
    if (w0 == u) return 0.0;
    int ru = rid[u], rv = rid[w0];
    if (ru == rv) return 0.0;                    /* inter-route only */

    int *B0 = w->bufC, L2 = 0;                   /* target route, flattened */
    {
        int vd = VD(rv);
        for (int c = nxt[vd]; c != vd; c = nxt[c]) B0[L2++] = c;
    }
    if (L2 < 1) return 0.0;
    int v;
    {
        int c1 = B0[rng_idx(rng, L2)], c2 = B0[rng_idx(rng, L2)];
        double s1 = w->inc[c1] - (K > 0 ? w->lb[c1] : 0.0);
        double s2 = w->inc[c2] - (K > 0 ? w->lb[c2] : 0.0);
        v = (s1 > s2) ? c1 : c2;
    }
    const double du = in->dem[u], dv = in->dem[v];
    if (S->load[ru] - du + dv > in->cap + EPS) return 0.0;
    if (S->load[rv] - dv + du > in->cap + EPS) return 0.0;

    double gu = w->inc[u] - dxy(w, prv[u], nxt[u]);      /* removal gains */
    double gv = w->inc[v] - dxy(w, prv[v], nxt[v]);

    double bi_v = 1e300; int aV = 0;             /* v into u's route, minus u */
    {
        int vd = VD(ru), a = vd;
        for (int c0 = nxt[vd]; ; c0 = nxt[c0]) {
            int b = (c0 == vd) ? vd : c0;
            if (b != u) {
                double ab = (nxt[a] != b) ? dxy(w, a, b) : (a <= n) ? w->bnx[a] : (b <= n) ? w->d0[b] : 0.0;
                double c = dxy(w, a, v) + dxy(w, v, b) - ab;
                if (c < bi_v) { bi_v = c; aV = a; }
                a = b;
            }
            if (c0 == vd) break;
        }
    }
    double bi_u = 1e300; int aU = 0;             /* u into v's route, minus v */
    {
        int vd = VD(rv), a = vd;
        for (int i2 = 0; ; i2++) {
            int b = (i2 == L2) ? vd : B0[i2];
            if (b != v) {
                double ab = (nxt[a] != b) ? dxy(w, a, b) : (a <= n) ? w->bnx[a] : (b <= n) ? w->d0[b] : 0.0;
                double c = dxy(w, a, u) + dxy(w, u, b) - ab;
                if (c < bi_u) { bi_u = c; aU = a; }
                a = b;
            }
            if (i2 == L2) break;
        }
    }
    double delta = bi_v + bi_u - gu - gv;
    if (probe) return delta;
    if (!sa_accept(delta, T, rng)) return 0.0;
    w->acc++;

    int pu = prv[u], qu = nxt[u];
    int pv = prv[v], qv = nxt[v];
    nxt[pu] = qu; prv[qu] = pu;                  /* detach u */
    nxt[pv] = qv; prv[qv] = pv;                  /* detach v */
    int bV = nxt[aV], bU = nxt[aU];              /* read after the removals */
    nxt[aV] = v; prv[v] = aV; nxt[v] = bV; prv[bV] = v;
    nxt[aU] = u; prv[u] = aU; nxt[u] = bU; prv[bU] = u;
    rid[u] = rv; rid[v] = ru;
    {   /* both loads as fresh sums (free here: the move is O(L1 + L2)) */
        double sl = 0.0; int vd = VD(ru);
        for (int c = nxt[vd]; c != vd; c = nxt[c]) sl += in->dem[c];
        S->load[ru] = sl;
        sl = 0.0; vd = VD(rv);
        for (int c = nxt[vd]; c != vd; c = nxt[c]) sl += in->dem[c];
        S->load[rv] = sl;
    }
    inc_upd(w, S, pu); inc_upd(w, S, qu);
    inc_upd(w, S, aV); inc_upd(w, S, bV);
    inc_upd(w, S, pv); inc_upd(w, S, qv);
    inc_upd(w, S, aU); inc_upd(w, S, bU);
    inc_upd(w, S, u);  inc_upd(w, S, v);
    return delta;
}

/* --------------------------------------------------------------- OPEN
 * u alone in an empty route (an emptied one first, a new one otherwise): the
 * only move that can raise the number of routes. */
static double mv_open(const Inst *in, WS *w, Sol *S, double T, Rng *rng, int probe)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;
    int u = pick_u(S, rng);
    int p = prv[u], q = nxt[u];
    if (p == q && p > n) return 0.0;            /* u is already alone */

    double delta = 2.0 * w->d0[u] + dxy(w, p, q) - w->inc[u];
    if (probe) return delta;
    if (!sa_accept(delta, T, rng)) return 0.0;

    int r = -1;
    for (int t = 0; t < S->R; t++)
        if (nxt[VD(t)] == VD(t)) { r = t; break; }
    if (r < 0) {
        if (S->R >= n) return 0.0;
        r = S->R++;
        int vd = VD(r);
        nxt[vd] = prv[vd] = vd; rid[vd] = r; S->load[r] = 0.0;
    }
    w->acc++;
    int ru = rid[u];
    nxt[p] = q; prv[q] = p;
    S->load[ru] -= in->dem[u];
    int vd = VD(r);
    nxt[vd] = u; prv[u] = vd; nxt[u] = vd; prv[vd] = u;
    rid[u] = r; S->load[r] = in->dem[u];
    inc_upd(w, S, p); inc_upd(w, S, q); inc_upd(w, S, u);
    return delta;
}

/* ---------------------------------------------------------------- 2-OPT
 * Same route: segment reversal. Different routes: 2-opt*, the tails of the
 * two routes are exchanged (the only operator that tests the capacity after
 * the Metropolis test). */
static HOT double mv_2opt(const Inst *in, WS *w, Sol *S, int K,
                          double T, Rng *rng, int probe)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;
    int u = pick_u(S, rng);
    int v = sa_cand(w, n, K, rng, u);
    if (v == u) return 0.0;
    int a = rng_bit(rng) ? prv[u] : u;        /* edge (a, nxt[a]) */
    int b = rng_bit(rng) ? prv[v] : v;        /* edge (b, nxt[b]) */
    if (a == b) return 0.0;
    int ra = rid[a], rb = rid[b];

    if (ra == rb) {                                  /* ---- intra 2-opt ---- */
        int sa = nxt[a], sb = nxt[b];
        if (sa == b || sb == a) return 0.0;
        double delta = dxy(w, a, b) + dxy(w, sa, sb) - dnx(w, S, a) - dnx(w, S, b);
        if (probe) return delta;
        if (!sa_accept(delta, T, rng)) return 0.0;
        w->acc++;
        /* reversing nxt[a]..b or its complement gives the same cycle: the
           order of a and b need not be known, and the segment may hold the
           virtual depot */
        int *buf = w->bufA, L = 0;
        for (int t = sa; ; t = nxt[t]) { buf[L++] = t; if (t == b) break; }
        int p = a;
        for (int t = L - 1; t >= 0; t--) {
            nxt[p] = buf[t]; prv[buf[t]] = p;
            if (p >= 1 && p <= n) w->bnx[p] = dxy(w, p, buf[t]);
            p = buf[t];
        }
        nxt[p] = sb; prv[sb] = p;
        if (p >= 1 && p <= n) w->bnx[p] = dxy(w, p, sb);
        inc_upd(w, S, a); inc_upd(w, S, sa);
        inc_upd(w, S, b); inc_upd(w, S, sb);
        return delta;
    }

    int sa = nxt[a], sb = nxt[b];                    /* ---- 2-opt* ---- */
    double delta = dxy(w, a, sb) + dxy(w, b, sa) - dnx(w, S, a) - dnx(w, S, b);
    if (!probe && !sa_accept(delta, T, rng)) return 0.0;

    int *A = w->bufA, *B = w->bufB, *NA = w->bufC, *NB = w->bufD;
    int LA = route_get(S, ra, A), LB = route_get(S, rb, B);
    int ia = -1, ib = -1;
    for (int t = 0; t < LA; t++) if (A[t] == a) { ia = t; break; }
    for (int t = 0; t < LB; t++) if (B[t] == b) { ib = t; break; }
    double prefA = 0, prefB = 0;
    for (int t = 0; t <= ia; t++) prefA += in->dem[A[t]];
    for (int t = 0; t <= ib; t++) prefB += in->dem[B[t]];
    double tailA = S->load[ra] - prefA, tailB = S->load[rb] - prefB;
    if (prefA + tailB > in->cap + EPS || prefB + tailA > in->cap + EPS) return 0.0;
    if (probe) return delta;
    w->acc++;

    int lna = 0, lnb = 0;
    for (int t = 0; t <= ia; t++)      NA[lna++] = A[t];
    for (int t = ib + 1; t < LB; t++)  NA[lna++] = B[t];
    for (int t = 0; t <= ib; t++)      NB[lnb++] = B[t];
    for (int t = ia + 1; t < LA; t++)  NB[lnb++] = A[t];
    route_set(S, in, ra, NA, lna);
    route_set(S, in, rb, NB, lnb);
    inc_upd(w, S, a); inc_upd(w, S, sa);
    inc_upd(w, S, b); inc_upd(w, S, sb);
    return delta;
}

/* ------------------------------------------------------ RUIN & RECREATE
 * The kick, a cut-down SISR (Christiaens & Vanden Berghe, Transp. Sci. 2020):
 * 1 to KICK_MAX customers are removed -- the string of a uniform seed, then
 * strings around its nearest neighbours -- and reinserted one by one, in the
 * order of removal, at their cheapest feasible position after one of their K
 * nearest neighbours or after their former predecessor. The whole is one
 * move. The Metropolis threshold is drawn first and the reconstruction is
 * abandoned as soon as it cannot be met any more (an insertion never costs
 * less than -slack). A rejected kick is undone exactly by replaying the
 * journal of its unlinks and links backwards. */
#define KICK_JMAX 64

COLD static double mv_kick(const Inst *in, WS *w, Sol *S, int K, double T, Rng *rng)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;
    if (n < 3) return 0.0;

    int jc[KICK_JMAX], jp[KICK_JMAX], jq[KICK_JMAX], jr[KICK_JMAX];
    char jins[KICK_JMAX];
    int rc[KICK_MAX], rp[KICK_MAX];              /* the removed, in order */
    int nj = 0, nrem = 0;
    double delta = 0.0;
    const int want = 1 + rng_idx(rng, KICK_MAX);
    if (++w->kick_gen == 0) {                    /* wrapped: 0 means "present" */
        memset(w->ktag, 0, (size_t)(n + 2) * sizeof(uint32_t));
        w->kick_gen = 1;
    }
    const uint32_t gen = w->kick_gen;

    /* --- ruin (inc[] goes stale here: lengths are read from the links) --- */
    int seed = pick_u(S, rng);
    for (int att = 0; att < 4 * KICK_MAX && nrem < want; att++) {
        int c = (att == 0) ? seed : sa_cand(w, n, K, rng, seed);
        int len = 1 + rng_idx(rng, want - nrem);
        for (int t = 0; t < len && nrem < want; t++) {
            if (c < 1 || c > n || w->ktag[c] == gen) break;
            int p = prv[c], q = nxt[c], r = rid[c];
            delta += dxy(w, p, q) - dxy(w, p, c) - dxy(w, c, q);
            nxt[p] = q; prv[q] = p;
            S->load[r] -= in->dem[c];
            w->ktag[c] = gen;
            jc[nj] = c; jp[nj] = p; jq[nj] = q; jr[nj] = r; jins[nj++] = 0;
            rc[nrem] = c; rp[nrem] = p; nrem++;
            c = q;
        }
    }
    if (!nrem) return 0.0;

    /* --- recreate --- */
    int ok = 1;
    const int ncand = (K > 0) ? K : 8;
    const double slack = in->rounded ? 1.0 : 1e-9;
    const double thr = (T > 0.0) ? -T * log(1.0 - rng_unit(rng)) : 0.0;
    for (int t = 0; t < nrem && ok; t++) {
        int c = rc[t];
        double bd = 1e300;
        int ba = -1;
        for (int e = -1; e < ncand; e++) {
            /* e = -1: the former predecessor */
            int v = (e < 0) ? rp[t]
                  : (K > 0) ? w->nbr[(size_t)(c - 1) * K + e]
                            : 1 + rng_idx(rng, n);
            if (v < 1) continue;
            if (v <= n && w->ktag[v] == gen) continue;     /* itself removed */
            int r = rid[v];
            if (r < 0 || r >= S->R) continue;
            if (S->load[r] + in->dem[c] > in->cap + EPS) continue;
            int b = nxt[v];
            double d = dxy(w, v, c) + dxy(w, c, b) - dxy(w, v, b);
            if (d < bd) { bd = d; ba = v; }
        }
        if (ba < 0) { ok = 0; break; }           /* nowhere feasible */
        int b = nxt[ba], r = rid[ba];
        nxt[ba] = c; prv[c] = ba; nxt[c] = b; prv[b] = c;
        rid[c] = r; S->load[r] += in->dem[c];
        w->ktag[c] = 0;
        jc[nj] = c; jp[nj] = ba; jq[nj] = b; jr[nj] = r; jins[nj++] = 1;
        delta += bd;
        if (delta - slack * (nrem - 1 - t) > thr) { ok = 0; break; }   /* lost */
    }

    /* --- accept, or roll back exactly --- */
    if (!ok || delta > thr) {
        for (int t = nj - 1; t >= 0; t--) {
            int c = jc[t];
            if (jins[t]) {
                int p = prv[c], q = nxt[c];
                nxt[p] = q; prv[q] = p;
                S->load[jr[t]] -= in->dem[c];
            } else {
                int p = jp[t], q = jq[t];
                nxt[p] = c; prv[c] = p; nxt[c] = q; prv[q] = c;
                rid[c] = jr[t]; S->load[jr[t]] += in->dem[c];
                w->ktag[c] = 0;
            }
        }
        return 0.0;
    }
    for (int t = 0; t < nrem; t++) {             /* refresh what moved */
        int c = jc[t];
        inc_upd(w, S, jp[t]); inc_upd(w, S, jq[t]);
        inc_upd(w, S, prv[c]); inc_upd(w, S, c); inc_upd(w, S, nxt[c]);
    }
    return delta;
}

/* --------------------------------------------------------- drawing a move */

static inline uint32_t op_thr(double c)        /* cumulative fraction -> uint32 */
{
    if (c >= 1.0) return 0xFFFFFFFFu;
    if (c <= 0.0) return 0u;
    return (uint32_t)(c * 4294967296.0);
}

static HOT double sa_draw(const Inst *in, WS *w, Sol *S, int K, double T,
                          Rng *rng, int probe)
{
    uint32_t z = rng32(rng);
    if (z < w->th1) return mv_relocate(in, w, S, K, T, rng, probe);
    if (z < w->th2) return mv_swap    (in, w, S, K, T, rng, probe);
    if (z < w->th3) return mv_2opt    (in, w, S, K, T, rng, probe);
    if (z < w->th5) return mv_swapstar(in, w, S, K, T, rng, probe);
    return                 mv_open    (in, w, S, T, rng, probe);
}

/* T0 = -mean(Delta+) / ln(CHI0), over up to 300 worsening moves drawn in probe
   mode from the start (Ben-Ameur 2004); 0 if no move worsens */
static double calibrate_T0(const Inst *in, WS *w, Sol *S, int K, Rng *rng)
{
    const int maxdraw = 2000, want = 300;
    double sum = 0.0;
    int ns = 0;
    for (int t = 0; t < maxdraw && ns < want; t++) {
        double d = sa_draw(in, w, S, K, 1.0, rng, 1);
        if (d > 0.0) { sum += d; ns++; }
    }
    if (!ns) return 0.0;
    return -(sum / ns) / log(CHI0);
}

/* One annealing chain: a start, its PRNG, its temperature and its best. */
typedef struct {
    Sol     sol;
    int    *b_nxt, *b_prv, *b_rid;  double *b_load;   /* best of the chain */
    Rng     rng;
    double  cur, best, T, alpha, T0, Tend, c0;
    int     bR, alive, frozen, visited;
    long    acc, spent;
    int     wall;                    /* --sa-wall: T follows the clock    */
    double  wall_end, frac;          /* deadline, fraction of time elapsed */
} Chain;

#define WALL_BUDGET (LONG_MAX / 4)   /* a step budget that never binds */

/* operator thresholds, kNN lists and bounds of the instance; returns K */
static int sa_config(const Inst *in, WS *w, int n)
{
    int K = K_SA; if (K > n - 1) K = n - 1; if (K < 0) K = 0;
    {   double wt = W_RELOCATE + W_SWAP + W_2OPT + W_SWAPSTAR + W_OPEN;
        double c1 = W_RELOCATE / wt, c2 = c1 + W_SWAP / wt;
        double c3 = c2 + W_2OPT / wt, c5 = c3 + W_SWAPSTAR / wt;
        w->th1 = op_thr(c1); w->th2 = op_thr(c2); w->th3 = op_thr(c3);
        w->th5 = op_thr(c5);
    }
    if (!w->xy_ok) { xy_build(in, w); w->xy_ok = 1; }
    if (K > 0) {
        knn_need(in, w, K);
        if (!w->lb_ok) { lb_build(in, w, K); w->lb_ok = 1; }
    }
    return K;
}

static void chain_init(const Inst *in, WS *w, Chain *c, int K, long budget, uint64_t seed,
                       int wall)
{
    const int n = c->sol.n;
    const size_t zi = (size_t)(2 * n + 4) * sizeof(int);
    const size_t zd = (size_t)(n + 2) * sizeof(double);
    inc_build(in, w, &c->sol);
    rng_seed(&c->rng, seed);
    c->best = c->cur; c->bR = c->sol.R;
    c->acc = 0; c->spent = 0; c->alive = 1; c->frozen = 0;
    c->wall = wall;

    double T0 = calibrate_T0(in, w, &c->sol, K, &c->rng);
    if (T0 <= 0.0) { c->alive = 0; c->frozen = 1; c->T0 = 0.0; return; }
    double Tend = T0 * pow(10.0, -DECADES);
    c->T = c->T0 = T0;
    c->Tend = Tend;
    c->alpha = (budget > 1) ? pow(Tend / T0, 1.0 / (double)(budget - 1)) : 1.0;
    c->frac = 0.0;
    if (c->wall) c->alpha = 1.0;               /* T is set from the clock */
    memcpy(c->b_nxt, c->sol.nxt, zi); memcpy(c->b_prv, c->sol.prv, zi);
    memcpy(c->b_rid, c->sol.rid, zi); memcpy(c->b_load, c->sol.load, zd);
}

/* --sa-wall: geometric in time, from T0 at the start to Tend at the deadline */
COLD static void chain_wall_temp(Chain *c)
{
    c->T = c->T0 * pow(c->Tend / c->T0, c->frac);
}

static HOT void chain_step(const Inst *in, WS *w, Chain *c, int K, long it,
                           size_t zi, size_t zd)
{
    c->cur += sa_draw(in, w, &c->sol, K, c->T, &c->rng, 0);
    if (((it + 1) & (LOAD_RESYNC - 1)) == 0) loads_resync(in, &c->sol);
    if ((it + 1) % KICK_EVERY == 0)
        c->cur += mv_kick(in, w, &c->sol, K, c->T, &c->rng);
    if (c->cur < c->best - 1e-12) {
        c->best = c->cur;
        c->bR = c->sol.R;
        memcpy(c->b_nxt, c->sol.nxt, zi); memcpy(c->b_prv, c->sol.prv, zi);
        memcpy(c->b_rid, c->sol.rid, zi); memcpy(c->b_load, c->sol.load, zd);
    }
    c->T *= c->alpha;
}

/* runs the chain over `budget` steps, or until its deadline under --sa-wall
   (the clock is read every 1024 steps, and T recomputed from it) */
static void anneal(const Inst *in, WS *w, Chain *c, int K, long budget)
{
    const int n = c->sol.n;
    const size_t zi = (size_t)(2 * n + 4) * sizeof(int);
    const size_t zd = (size_t)(n + 2) * sizeof(double);
    if (!c->alive) return;
    w->acc = c->acc;
    if (c->wall) {
        const double tb = now_sec(), len = c->wall_end - tb;
        long it = 0;
        if (len > 0.0)
            for (;;) {
                chain_step(in, w, c, K, it, zi, zd);
                if ((++it & 1023) == 0) {
                    const double t = now_sec();
                    c->frac = (t - tb) / len;
                    if (w->tr_on && c->best < w->tr_best) trace_note(w, t, c->best);
                    if (c->frac >= 1.0) break;
                    chain_wall_temp(c);
                }
            }
        c->acc = w->acc;
        c->spent = it;
        return;
    }
    for (long it = 0; it < budget; it++) chain_step(in, w, c, K, it, zi, zd);
    c->acc = w->acc;
    c->spent = budget;
}

/* restores the best of the chain into its solution */
static double chain_finish(const Inst *in, WS *w, Chain *c)
{
    const int n = c->sol.n;
    const size_t zi = (size_t)(2 * n + 4) * sizeof(int);
    const size_t zd = (size_t)(n + 2) * sizeof(double);
    if (!c->frozen) {
        memcpy(c->sol.nxt, c->b_nxt, zi); memcpy(c->sol.prv, c->b_prv, zi);
        memcpy(c->sol.rid, c->b_rid, zi); memcpy(c->sol.load, c->b_load, zd);
        c->sol.R = c->bR;
        inc_build(in, w, &c->sol);
    }
    return c->best;
}

/* =========================================================================
 *  One instance: RESTARTS starts, each built and annealed, best kept.
 * ========================================================================= */

typedef struct {
    double cost, cost0, mean0, mean1, acc, t0;
    long   steps;          /* steps per start (--sa-wall: mean run per start) */
    int    routes, feasible;
    double time_ms;
    double *trace; int trace_n;   /* --trace: 2 * trace_n values (t, cost) */
} Result;

typedef struct {
    const char *dir, *sol;
    int    rounded, threads, quiet, trace;
    uint64_t seed;
    long   sa_steps;       /* per start */
    double sa_wall;        /* seconds per instance, all starts together */
} Opts;

static void solve(const Inst *in, WS *w, const Opts *o, Result *res,
                  int **sol_out, uint64_t seed)
{
    const int n = in->n;
    const double t_start = now_sec();
    const int wall = (o->sa_wall > 0.0);
    if ((unsigned)n > MAXN) die("n=%d exceeds %u", n, MAXN);
    w->knn_k = 0; w->xy_ok = 0; w->lb_ok = 0;    /* per-instance caches */
    w->tr_on = o->trace; w->tr_n = 0; w->tr_t0 = t_start;
    w->tr_best = 1e300; w->tr_open = 0.0; w->tr_res = 1e-4 * o->sa_wall;

    /* savings: exact up to CW_EXACT_N customers, on the CW_KNN nearest beyond */
    const int exact = (n <= CW_EXACT_N);
    int Kcw = exact ? n - 1 : CW_KNN;
    if (Kcw > n - 1) Kcw = n - 1;
    if (Kcw < 1) Kcw = 1;
    size_t nsav = exact ? ((size_t)n * (size_t)(n - 1)) / 2 : (size_t)n * (size_t)Kcw;
    ws_ensure(w, n, exact ? K_SA : Kcw, nsav ? nsav : 1);
    for (int i = 0; i <= n; i++) w->d0[i] = dist(in, 0, i);

    const size_t zi = (size_t)(2 * n + 4) * sizeof(int);
    const size_t zd = (size_t)(n + 2) * sizeof(double);
    Chain c; memset(&c, 0, sizeof c);
    c.sol.n = n;
    c.sol.nxt = w->s_nxt; c.sol.prv = w->s_prv; c.sol.rid = w->s_rid; c.sol.load = w->s_load;
    c.b_nxt = w->b_nxt; c.b_prv = w->b_prv; c.b_rid = w->b_rid; c.b_load = w->b_load;

    double bestc = 1e300, bestc0 = 0, sum0 = 0, sum1 = 0;
    int bestR = 0, best_visited = 0;
    long spent_sum = 0;
    res->acc = res->t0 = 0.0;
    const long budget = wall ? WALL_BUDGET : o->sa_steps;

    for (int rs = 0; rs < RESTARTS; rs++) {
        c.c0 = build_cw_start(in, w, &c.sol, rs, seed, exact, Kcw, &c.visited);
        c.cur = c.c0;
        if (w->tr_on && c.visited == n) trace_note(w, now_sec(), c.c0);
        double cost = c.c0;
        double acc = 0.0, t0 = 0.0;
        if (o->sa_steps > 0) {
            int K = sa_config(in, w, n);
            chain_init(in, w, &c, K, budget,
                       (seed ^ 0x5DEECE66DULL) + (uint64_t)rs * 0xBF58476D1CE4E5B9ULL, wall);
            if (wall) {                          /* the time left, shared */
                const double left = o->sa_wall - (now_sec() - t_start);
                c.wall_end = now_sec() + left / (double)(RESTARTS - rs);
            }
            anneal(in, w, &c, K, budget);
            chain_finish(in, w, &c);
            cost = sol_cost(in, &c.sol);
            acc = c.spent ? (double)c.acc / (double)c.spent : 0.0;
            t0 = c.T0;
        }
        if (w->tr_on) trace_note(w, now_sec(), cost);
        sum0 += c.c0; sum1 += cost;
        if (cost < bestc) {
            bestc = cost; bestc0 = c.c0; bestR = c.sol.R; best_visited = c.visited;
            res->acc = acc; res->t0 = t0;
            memcpy(w->r_nxt, c.sol.nxt, zi); memcpy(w->r_prv, c.sol.prv, zi);
            memcpy(w->r_rid, c.sol.rid, zi); memcpy(w->r_load, c.sol.load, zd);
        }
        spent_sum += c.spent;
    }

    Sol sol = c.sol;
    memcpy(sol.nxt, w->r_nxt, zi); memcpy(sol.prv, w->r_prv, zi);
    memcpy(sol.rid, w->r_rid, zi); memcpy(sol.load, w->r_load, zd);
    sol.R = bestR;
    res->steps = wall ? spent_sum / RESTARTS : o->sa_steps;
    res->trace = NULL; res->trace_n = 0;
    if (w->tr_on) {
        trace_note(w, now_sec(), bestc);
        /* the last entry is the returned solution: absorb the rounding drift
           between the incrementally tracked cost and the recomputed one */
        if (w->tr_n > 0 && fabs(w->tr_buf[2 * w->tr_n - 1] - bestc) <= 1e-6 * fmax(1.0, bestc))
            w->tr_buf[2 * w->tr_n - 1] = bestc;
        res->trace = w->tr_buf; res->trace_n = w->tr_n;
        w->tr_buf = NULL; w->tr_n = w->tr_cap = 0;
    }
    res->mean0 = sum0 / RESTARTS;
    res->mean1 = sum1 / RESTARTS;

    /* --- checking, and the routes as [n_words, c.., 0, c.., 0] --- */
    int *seen = w->seen, *rt = w->route;
    int *ob = NULL, obn = 1;
    if (sol_out) ob = (int *)xmalloc((size_t)(n + bestR + 3) * sizeof(int));
    int nroutes = 0, feasible = (best_visited == n);
    for (int i = 1; i <= n; i++) seen[i] = 0;
    for (int r = 0; r < bestR; r++) {
        int vd = n + 1 + r, L = 0;
        double rload = 0.0;
        for (int t = sol.nxt[vd]; t != vd; t = sol.nxt[t]) {
            if (t < 1 || t > n || seen[t]) { feasible = 0; break; }
            seen[t] = 1; rload += in->dem[t]; rt[++L] = t;
        }
        if (!L) continue;                            /* emptied by annealing */
        nroutes++;
        if (rload > in->cap + EPS) feasible = 0;
        if (ob) {
            for (int t = 1; t <= L; t++) ob[obn++] = rt[t];
            ob[obn++] = 0;
        }
    }
    for (int i = 1; i <= n; i++) if (!seen[i]) feasible = 0;
    if (ob) { ob[0] = obn - 1; *sol_out = ob; }
    res->cost = bestc; res->cost0 = bestc0;
    res->routes = nroutes; res->feasible = feasible;
}

/* ============================================================== main / CLI */

static void usage(void)
{
    printf(
"SAVANT for the CVRP, minimal version (the configuration of the paper)\n\n"
"Usage: savant --dir DIR [options]\n\n"
"  --dir DIR        solve every .vrp instance of DIR (CVRPLIB format, EUC_2D)\n"
"  --round          round distances to the nearest integer (CVRPLIB convention)\n"
"  --sa-steps S     annealing steps per start (%d starts; default 1000)\n"
"  --sa-wall T      instead: T seconds per instance, shared by the starts,\n"
"                   temperature driven by the elapsed time\n"
"  --seed S         random seed (default 42)\n"
"  --sol FILE       write the solutions (the format of cw --sol)\n"
"  --trace          with --sa-wall and --sol: best cost vs elapsed time\n"
"  --threads T      instances solved in parallel (default: all cores)\n"
"  -q               print the mean cost only\n", RESTARTS);
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    Opts o;
    memset(&o, 0, sizeof o);
    o.seed = 42; o.sa_steps = 1000;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT() (i + 1 < argc ? argv[++i] : (die("option %s: missing argument", a), ""))
        if      (!strcmp(a, "--dir"))      o.dir = NEXT();
        else if (!strcmp(a, "--round"))    o.rounded = 1;
        else if (!strcmp(a, "--sa-steps")) o.sa_steps = opt_long(a, NEXT());
        else if (!strcmp(a, "--sa-wall"))  o.sa_wall = opt_num(a, NEXT());
        else if (!strcmp(a, "--seed"))     o.seed = opt_u64(a, NEXT());
        else if (!strcmp(a, "--sol"))      o.sol = NEXT();
        else if (!strcmp(a, "--trace"))    o.trace = 1;
        else if (!strcmp(a, "--threads"))  o.threads = (int)opt_long(a, NEXT());
        else if (!strcmp(a, "-q"))         o.quiet = 1;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else die("unknown option: %s (see --help)", a);
        #undef NEXT
    }
    if (!o.dir) { usage(); return 1; }
    if (o.sa_wall < 0.0) die("--sa-wall: a positive number of seconds");
    if (o.sa_wall > 0.0 && o.sa_steps <= 0) die("--sa-wall with --sa-steps 0: nothing to anneal");
    if (o.trace && (o.sa_wall <= 0.0 || !o.sol))
        die("--trace needs --sa-wall (the time axis) and --sol (where it goes)");

    int M;
    Inst *insts = read_dir(o.dir, o.rounded, &M);
    int nthreads = o.threads > 0 ? o.threads : omp_get_max_threads();
    omp_set_num_threads(nthreads);
    int **solbuf = o.sol ? (int **)xmalloc((size_t)M * sizeof(int *)) : NULL;
    Result *R = (Result *)xmalloc((size_t)M * sizeof(Result));
    double t0 = now_sec();

    #pragma omp parallel
    {
        WS w; memset(&w, 0, sizeof w);
        #pragma omp for schedule(dynamic, 1)
        for (int k = 0; k < M; k++) {
            double s = now_sec();
            solve(&insts[k], &w, &o, &R[k], solbuf ? &solbuf[k] : NULL,
                  o.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)k + 1);
            R[k].time_ms = (now_sec() - s) * 1e3;
        }
        ws_free(&w);
    }
    double wall = now_sec() - t0;

    if (solbuf) {
        FILE *f = fopen(o.sol, "w");
        if (!f) die("writing %s: %s", o.sol, strerror(errno));
        fprintf(f, "#CWSOL 1\n#instances %d\n", M);
        fprintf(f, "#source %s\n", o.dir);
        fprintf(f, "#round %d\n", o.rounded ? 1 : 0);
        if (o.sa_wall > 0.0) {
            fprintf(f, "#sa-wall %g\n", o.sa_wall);
            fprintf(f, "#sa-steps");
            for (int k = 0; k < M; k++) fprintf(f, " %ld", R[k].steps);
            fprintf(f, "\n");
        }
        fprintf(f, "#format inst <idx> <name> <n> <Q> <cost> <routes>, "
                   "then one line per route\n");
        if (o.trace)
            fprintf(f, "#format trace <idx> <elapsed s> <best cost>, "
                       "on each improvement\n");
        for (int k = 0; k < M; k++) {
            for (int j = 0; j < R[k].trace_n; j++)
                fprintf(f, "#trace %d %.6f %.17g\n", k,
                        R[k].trace[2 * j], R[k].trace[2 * j + 1]);
            free(R[k].trace); R[k].trace = NULL;
            fprintf(f, "inst %d %s %d %.17g %.17g %d\n", k, insts[k].name,
                    insts[k].n, insts[k].cap, R[k].cost, R[k].routes);
            const int *b = solbuf[k];
            int first = 1;
            for (int t = 1; t <= b[0]; t++) {
                if (b[t] == 0) { fputc('\n', f); first = 1; }
                else { fprintf(f, first ? "%d" : " %d", b[t]); first = 0; }
            }
            free(solbuf[k]);
        }
        if (ferror(f) || fclose(f)) die("writing %s: %s", o.sol, strerror(errno));
        free(solbuf);
    }

    double sum = 0, sum0 = 0, sumr = 0, cpu = 0, sacc = 0, sm1 = 0;
    int infeas = 0;
    double *all = (double *)xmalloc((size_t)M * sizeof(double));
    for (int k = 0; k < M; k++) {
        sum += R[k].cost; sum0 += R[k].cost0; sumr += R[k].routes; cpu += R[k].time_ms;
        sacc += R[k].acc; sm1 += R[k].mean1;
        if (!R[k].feasible) infeas++;
        all[k] = R[k].cost;
    }
    double mean = sum / M;
    qsort(all, (size_t)M, sizeof(double), cmp_double);
    if (!o.quiet) {
        printf("instances        : %d  (%d infeasible)\n", M, infeas);
        printf("mean cost of the best start, before annealing : %.5f\n", sum0 / M);
        printf("mean cost after annealing  : %.5f\n", mean);
        printf("mean over the %d starts    : %.5f\n", RESTARTS, sm1 / M);
        printf("acceptance rate  : %.1f %%\n", 100.0 * sacc / M);
        printf("median / min / max : %.5f / %.5f / %.5f\n", all[M / 2], all[0], all[M - 1]);
        printf("routes (mean)    : %.3f\n", sumr / M);
        printf("total time       : %.3f s (wall), %.3f s (cumulative CPU)\n", wall, cpu / 1e3);
    } else {
        printf("%.6f\n", mean);
    }
    free(all); free(R);
    for (int k = 0; k < M; k++) inst_free(&insts[k]);
    free(insts);
    return infeas ? 2 : 0;
}
