/* ============================================================================
 *  cw.c — Clarke & Wright savings heuristic ("parallel" version) for the
 *         CVRP.  High-performance C implementation.
 *
 *  Key points:
 *    - savings list truncated to the K nearest neighbours (uniform grid)
 *      => O(n K) instead of O(n^2), exact when K >= n-1
 *    - LSD radix sort, 4x8 bits, on the float key (savings > 0 => bit
 *      patterns are monotone), stable and deterministic
 *    - route merging via adjacency lists (degree <= 2) + union-find
 *      => no route reversal ever needed, O(alpha(n)) per merge
 *    - OpenMP parallelism at the instance level, per-thread buffers reused
 *      (zero malloc in the hot loop)
 *    - parameterised savings (Yellow): s_ij = d0i + d0j - lambda*dij
 *                                            + mu*|d0i - d0j|
 *    - optional intra-route 2-opt
 *
 *  Build:  make          (see Makefile)
 *  Help :  ./cw --help
 * ==========================================================================*/

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
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

#define MAXN 65535u          /* i,j packed into 16 bits each */
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

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* --------------------------------------------------------------- RNG (PRNG)
 * splitmix64 to seed, xoshiro256** for the stream. Every instance gets its
 * own seed (seed + index), so results are reproducible whatever the thread
 * scheduling. */

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

/* One 64-bit draw serves two 32-bit integers: the PRNG used to be ~25 % of
   the per-step cost, with 4 to 5 calls to xoshiro. */
static inline uint32_t rng32(Rng *r)
{
    if (r->has32) { r->has32 = 0; return (uint32_t)(r->w32 >> 32); }
    r->w32 = rng_next(r); r->has32 = 1;
    return (uint32_t)r->w32;
}

/* integer in [0,n) by multiply-shift (Lemire), no rejection loop */
static inline int rng_idx(Rng *r, int n)
{
    return (int)(((uint64_t)rng32(r) * (uint64_t)n) >> 32);
}

/* one bit drawn from a 64-bit reservoir: ~4 operations instead of ~10 */
static inline int rng_bit(Rng *r)
{
    if (r->nbits == 0) { r->pool = rng_next(r); r->nbits = 64; }
    int b = (int)(r->pool & 1);
    r->pool >>= 1; r->nbits--;
    return b;
}

static inline double rng_unit(Rng *r)          /* [0,1) */
{
    return (double)(rng_next(r) >> 11) * 0x1.0p-53;
}

static inline uint32_t rng_bounded(Rng *r, uint32_t bound)   /* [0,bound) */
{
    uint64_t m = (uint64_t)(uint32_t)rng_next(r) * (uint64_t)bound;
    uint32_t l = (uint32_t)m;
    if (l < bound) {
        uint32_t t = (uint32_t)(-(int32_t)bound) % bound;
        while (l < t) { m = (uint64_t)(uint32_t)rng_next(r) * (uint64_t)bound; l = (uint32_t)m; }
    }
    return (uint32_t)(m >> 32);
}

/* ----------------------------------------------------------------- options */

typedef struct {
    const char *dir, *bundle, *csv, *sol, *dump, *validate;
    int   random, n, m, threads, knn, do2opt, rounded, limit, quiet, per_inst;
    int   init_random;      /* --init random : skip the savings build */
    uint64_t seed;
    double cap, lambda, mu;
    /* simulated annealing */
    int    sa_steps, sa_knn, check, split, split_every, split_tour;
    int    restarts, cw_rand, pick_t, pick_crit;
    double pick_eps;
    double cw_alpha;
    double sa_t0, sa_tend, sa_chi0, sa_decades;
    double w_rel, w_swap, w_2opt, w_or;
    int    or_max;
} Opts;

/* ---------------------------------------------------------------- instance
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

/* ---------------------------------------------------------- random generation
 * Nazari/Kool distribution, as reused by NeuOpt:
 * coordinates U[0,1]^2, integer demands U{1..9}, fixed capacity. */

static double default_capacity(int n)
{
    if (n <= 20)  return 30.0;
    if (n <= 50)  return 40.0;
    if (n <= 100) return 50.0;
    if (n <= 200) return 70.0;
    if (n <= 500) return 100.0;
    return 200.0;
}

static void gen_instance(Inst *in, int n, double cap, uint64_t seed, int rounded)
{
    Rng r; rng_seed(&r, seed);
    in->n = n; in->cap = cap; in->rounded = rounded;
    in->x[0] = rng_unit(&r); in->y[0] = rng_unit(&r); in->dem[0] = 0.0;
    for (int i = 1; i <= n; i++) {
        in->x[i] = rng_unit(&r);
        in->y[i] = rng_unit(&r);
    }
    for (int i = 1; i <= n; i++) in->dem[i] = (double)(1 + rng_bounded(&r, 9));
    snprintf(in->name, sizeof in->name, "rnd-n%d-s%llu", n, (unsigned long long)seed);
}

/* ------------------------------------------------------------- bundle reader
 * Binary .cvrpb format (little-endian), produced by fetch_neuopt.py:
 *   char  magic[8] = "CVRPBIN1"
 *   u32   n_instances
 *   u32   reserved (0)
 *   for each instance:
 *     u32    n              (number of customers, depot excluded)
 *     f64    capacity
 *     f64    x[n+1], y[n+1] (index 0 = depot)
 *     f64    dem[n+1]       (dem[0] = 0)
 */

static Inst *read_bundle(const char *path, int limit, int rounded, int *count)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("opening %s: %s", path, strerror(errno));

    char magic[8];
    uint32_t nb, rsv;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "CVRPBIN1", 8) != 0)
        die("%s: invalid .cvrpb signature", path);
    if (fread(&nb, 4, 1, f) != 1 || fread(&rsv, 4, 1, f) != 1)
        die("%s: truncated header", path);

    int m = (int)nb;
    if (limit > 0 && limit < m) m = limit;

    Inst *v = (Inst *)xmalloc((size_t)m * sizeof(Inst));
    for (int k = 0; k < m; k++) {
        uint32_t n; double cap;
        if (fread(&n, 4, 1, f) != 1 || fread(&cap, 8, 1, f) != 1)
            die("%s: instance %d truncated", path, k);
        if (n < 1 || n > MAXN) die("%s: n=%u out of range (1..%u)", path, n, MAXN);
        inst_alloc(&v[k], (int)n);
        v[k].cap = cap; v[k].rounded = rounded;
        size_t sz = (size_t)n + 1;
        if (fread(v[k].x, 8, sz, f) != sz ||
            fread(v[k].y, 8, sz, f) != sz ||
            fread(v[k].dem, 8, sz, f) != sz)
            die("%s: instance %d truncated", path, k);
        const char *base = strrchr(path, '/'); base = base ? base + 1 : path;
        snprintf(v[k].name, sizeof v[k].name, "%.40s#%d", base, k);
    }
    fclose(f);
    *count = m;
    return v;
}

/* -------------------------------------------------------- TSPLIB/.vrp reader
 * Accepts classic CVRPLIB files (integer coordinates) as well as the
 * floating-point variant produced by fetch_neuopt.py --tsplib.
 * The depot is the one given by DEPOT_SECTION (node 1 by default). */

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
        } else if (!strncmp(p, "NODE_COORD_SECTION", 18)) {
            if (dim < 0) { fclose(f); return 0; }
            for (int k = 0; k < dim; k++) {
                int id; double a, b;
                if (fscanf(f, "%d %lf %lf", &id, &a, &b) != 3) { fclose(f); return 0; }
                if (id < 1 || id > dim) { fclose(f); return 0; }
                X[id] = a; Y[id] = b;
            }
        } else if (!strncmp(p, "DEMAND_SECTION", 14)) {
            for (int k = 0; k < dim; k++) {
                int id; double d;
                if (fscanf(f, "%d %lf", &id, &d) != 2) { fclose(f); return 0; }
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
}

/* ---------------------------------------------------------- directory reader */

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static int has_suffix(const char *s, const char *suf)
{
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && !strcmp(s + ls - lf, suf);
}

static Inst *read_dir(const char *dirpath, int limit, int rounded, int *count)
{
    DIR *d = opendir(dirpath);
    if (!d) die("opening directory %s: %s", dirpath, strerror(errno));

    char **names = NULL; int nn = 0, cap = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!(has_suffix(e->d_name, ".vrp") || has_suffix(e->d_name, ".txt") ||
              has_suffix(e->d_name, ".cvrpb"))) continue;
        if (nn == cap) { cap = cap ? 2 * cap : 64; names = (char **)xrealloc(names, (size_t)cap * sizeof(char *)); }
        names[nn++] = strdup(e->d_name);
    }
    closedir(d);
    if (!nn) die("no .vrp/.txt/.cvrpb file in %s", dirpath);
    qsort(names, (size_t)nn, sizeof(char *), cmp_str);

    Inst *v = NULL; int nv = 0, vcap = 0;
    char path[4096];
    for (int i = 0; i < nn && (limit <= 0 || nv < limit); i++) {
        snprintf(path, sizeof path, "%s/%s", dirpath, names[i]);
        if (has_suffix(names[i], ".cvrpb")) {
            int c = 0;
            int rem = (limit > 0) ? limit - nv : 0;
            Inst *sub = read_bundle(path, rem, rounded, &c);
            if (nv + c > vcap) { vcap = nv + c; v = (Inst *)xrealloc(v, (size_t)vcap * sizeof(Inst)); }
            memcpy(v + nv, sub, (size_t)c * sizeof(Inst));
            free(sub); nv += c;
        } else {
            Inst tmp;
            if (!read_vrp(path, &tmp, rounded)) {
                fprintf(stderr, "warning: %s skipped (unrecognised format)\n", path);
                continue;
            }
            if (nv == vcap) { vcap = vcap ? 2 * vcap : 64; v = (Inst *)xrealloc(v, (size_t)vcap * sizeof(Inst)); }
            v[nv++] = tmp;
        }
    }
    for (int i = 0; i < nn; i++) free(names[i]);
    free(names);
    if (!nv) die("no readable instance in %s", dirpath);
    *count = nv;
    return v;
}

/* =========================================================================
 *  Per-thread workspace: everything is allocated once, then reused.
 * ========================================================================= */

typedef struct { uint32_t key; uint32_t ij; } Sav;   /* ij = (i<<16)|j */

typedef struct {
    int cap_n, cap_k;
    size_t cap_sav, cap_nbr;
    /* geometry */
    double *d0;             /* distance depot -> i          (n+1) */
    int    *cell_start;     /* grid                               */
    int    *cell_pts;
    double *cell_xy;
    int     gx, gy;
    /* knn */
    int    *nbr;            /* n*K                                */
    double *hd; int *hid;   /* bounded heap of size K             */
    /* savings */
    Sav *sav, *tmp;
    /* merging */
    int    *uf, *deg, *adj; /* adj: 2*(n+1)                       */
    double *load;
    /* reconstruction / checking */
    int    *route, *seen;
    /* simulated annealing */
    int    *flat, *rstart;
    int    *s_nxt, *s_prv, *s_rid;   double *s_load;
    int    *b_nxt, *b_prv, *b_rid;   double *b_load;
    int    *r_nxt, *r_prv, *r_rid;   double *r_load;
    int    *bufA, *bufB, *bufC, *bufD;
    int    *tour, *spPred, *spDq;
    void   *spAng;
    double *spD, *spC, *spP, *spF;
    long    acc;
    int     knn_k, pick_t;
    double  t0_used, gsplit;
    double *xy;
    int     rounded;
    double *lb, *bad, *inc, *fen, *fen_wv, fen_tot, eps0;
    int     fen_pw, crit, or_max;
    uint32_t th1, th2, th3;
    Inst    scratch;        /* instance generated on the fly      */
    int     scratch_cap;
} WS;

static void ws_ensure(WS *w, int n, int K, size_t nsav)
{
    if (n > w->cap_n) {
        int c = n;
        w->d0   = (double *)xrealloc(w->d0,   (size_t)(c + 1) * sizeof(double));
        w->uf   = (int *)   xrealloc(w->uf,   (size_t)(c + 1) * sizeof(int));
        w->deg  = (int *)   xrealloc(w->deg,  (size_t)(c + 1) * sizeof(int));
        w->adj  = (int *)   xrealloc(w->adj,  (size_t)2 * (c + 1) * sizeof(int));
        w->load = (double *)xrealloc(w->load, (size_t)(c + 1) * sizeof(double));
        w->route= (int *)   xrealloc(w->route,(size_t)(c + 2) * sizeof(int));
        w->seen = (int *)   xrealloc(w->seen, (size_t)(c + 1) * sizeof(int));
        w->cell_start = (int *)xrealloc(w->cell_start, (size_t)(4 * c + 16) * sizeof(int));
        w->cell_pts   = (int *)xrealloc(w->cell_pts,   (size_t)(c + 1) * sizeof(int));
        w->cell_xy    = (double *)xrealloc(w->cell_xy, (size_t)(2 * c + 4) * sizeof(double));
        size_t z = (size_t)2 * c + 4;               /* customers + virtual depots */
        w->flat   = (int *)xrealloc(w->flat,   (size_t)(c + 2) * sizeof(int));
        w->rstart = (int *)xrealloc(w->rstart, (size_t)(c + 2) * sizeof(int));
        w->s_nxt  = (int *)xrealloc(w->s_nxt,  z * sizeof(int));
        w->s_prv  = (int *)xrealloc(w->s_prv,  z * sizeof(int));
        w->s_rid  = (int *)xrealloc(w->s_rid,  z * sizeof(int));
        w->b_nxt  = (int *)xrealloc(w->b_nxt,  z * sizeof(int));
        w->b_prv  = (int *)xrealloc(w->b_prv,  z * sizeof(int));
        w->b_rid  = (int *)xrealloc(w->b_rid,  z * sizeof(int));
        w->s_load = (double *)xrealloc(w->s_load, (size_t)(c + 2) * sizeof(double));
        w->b_load = (double *)xrealloc(w->b_load, (size_t)(c + 2) * sizeof(double));
        w->r_nxt  = (int *)xrealloc(w->r_nxt, z * sizeof(int));
        w->r_prv  = (int *)xrealloc(w->r_prv, z * sizeof(int));
        w->r_rid  = (int *)xrealloc(w->r_rid, z * sizeof(int));
        w->r_load = (double *)xrealloc(w->r_load, (size_t)(c + 2) * sizeof(double));
        w->lb     = (double *)xrealloc(w->lb,     (size_t)(c + 2) * sizeof(double));
        w->bad    = (double *)xrealloc(w->bad,    (size_t)(c + 2) * sizeof(double));
        w->xy     = (double *)xrealloc(w->xy, (size_t)(4 * c + 12) * sizeof(double));
        w->inc    = (double *)xrealloc(w->inc,    (size_t)(c + 2) * sizeof(double));
        w->fen    = (double *)xrealloc(w->fen,    (size_t)(c + 2) * sizeof(double));
        w->fen_wv = (double *)xrealloc(w->fen_wv, (size_t)(c + 2) * sizeof(double));
        w->bufA = (int *)xrealloc(w->bufA, (size_t)(c + 2) * sizeof(int));
        w->bufB = (int *)xrealloc(w->bufB, (size_t)(c + 2) * sizeof(int));
        w->bufC = (int *)xrealloc(w->bufC, (size_t)(c + 2) * sizeof(int));
        w->bufD = (int *)xrealloc(w->bufD, (size_t)(c + 2) * sizeof(int));
        w->tour   = (int *)xrealloc(w->tour,   (size_t)(c + 2) * sizeof(int));
        w->spPred = (int *)xrealloc(w->spPred, (size_t)(c + 2) * sizeof(int));
        w->spDq   = (int *)xrealloc(w->spDq,   (size_t)(c + 2) * sizeof(int));
        w->spD = (double *)xrealloc(w->spD, (size_t)(c + 2) * sizeof(double));
        w->spC = (double *)xrealloc(w->spC, (size_t)(c + 2) * sizeof(double));
        w->spP = (double *)xrealloc(w->spP, (size_t)(c + 2) * sizeof(double));
        w->spF = (double *)xrealloc(w->spF, (size_t)(c + 2) * sizeof(double));
        w->spAng = xrealloc(w->spAng, (size_t)(c + 2) * (sizeof(double) + sizeof(int) + 8));
        w->cap_n = c;
    }
    size_t need = (size_t)n * (size_t)K;
    if (need > w->cap_nbr) {
        w->nbr = (int *)xrealloc(w->nbr, need * sizeof(int));
        w->cap_nbr = need;
    }
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
    if (n > w->scratch_cap) {
        inst_free(&w->scratch);
        inst_alloc(&w->scratch, n);
        w->scratch_cap = n;
    }
}

static void ws_free(WS *w)
{
    free(w->d0); free(w->uf); free(w->deg); free(w->adj); free(w->load);
    free(w->route); free(w->seen); free(w->cell_start); free(w->cell_pts); free(w->cell_xy);
    free(w->nbr); free(w->hd); free(w->hid); free(w->sav); free(w->tmp);
    free(w->flat); free(w->rstart);
    free(w->s_nxt); free(w->s_prv); free(w->s_rid); free(w->s_load);
    free(w->b_nxt); free(w->b_prv); free(w->b_rid); free(w->b_load);
    free(w->r_nxt); free(w->r_prv); free(w->r_rid); free(w->r_load); free(w->xy); free(w->lb); free(w->bad); free(w->inc); free(w->fen); free(w->fen_wv);
    free(w->bufA); free(w->bufB); free(w->bufC); free(w->bufD);
    free(w->tour); free(w->spPred); free(w->spDq);
    free(w->spD); free(w->spC); free(w->spP); free(w->spF); free(w->spAng);
    if (w->scratch_cap) inst_free(&w->scratch);
}

/* --------------------------------------------------------------- union-find */

static inline int uf_find(int *uf, int i)
{
    int r = i;
    while (uf[r] != r) r = uf[r];
    while (uf[i] != r) { int t = uf[i]; uf[i] = r; i = t; }
    return r;
}

/* ------------------------------------------------------------ LSD radix sort
 * uint32 keys coming from float >= 0: the order of the bit patterns matches
 * the numeric order. Stable sort => deterministic result. */

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

/* ------------------------------------------------------- k nearest neighbours
 * Uniform grid scanned in rings (the points live in a square); brute force
 * below a threshold (less overhead). */

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
            int *out = w->nbr + (size_t)(i - 1) * K;
            for (int t = 0; t < K; t++) out[t] = (t < sz) ? w->hid[t] : 0;
        }
        return;
    }

    /* --- build the grid --- */
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

    int *start = w->cell_start;                 /* size >= ncell+1 */
    int *pts   = w->cell_pts;
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
    /* Coordinates are copied out in cell order, so scanning a ring becomes
       sequential in memory. Without this, every candidate cost two random
       accesses into x[] and y[], which dominated build time on large
       instances. */
    for (int t = 0; t < n; t++) { int j = pts[t]; pxy[2 * t] = in->x[j]; pxy[2 * t + 1] = in->y[j]; }

    const double step = (cw < ch) ? cw : ch;
    const int maxr = (gx > gy ? gx : gy);

    /* Queries are processed in cell order rather than index order: two
       consecutive queries are then spatially close and re-read the same
       cells, which stay in cache. On large instances this was the single
       most expensive effect. */
    double *hd = w->hd; int *hid = w->hid;
    for (int qi = 0; qi < n; qi++) {
        const int i = pts[qi];
        int sz = 0;
        double worst = 1e300;                    /* keep the threshold in a register */
        const int cx = CELLX(i), cy = CELLY(i);
        const double xi = in->x[i], yi = in->y[i];
        for (int r = 0; r <= maxr; r++) {
            int x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r;
            if (x0 < 0 && x1 >= gx && y0 < 0 && y1 >= gy && r > 0) break;
            for (int vy = (y0 < 0 ? 0 : y0); vy <= (y1 >= gy ? gy - 1 : y1); vy++) {
                int onEdgeY = (vy == y0 || vy == y1);
                for (int vx = (x0 < 0 ? 0 : x0); vx <= (x1 >= gx ? gx - 1 : x1); vx++) {
                    if (!onEdgeY && vx != x0 && vx != x1) { /* skip the interior */
                        if (x1 < gx) vx = x1 - 1; else vx = gx - 1;
                        continue;
                    }
                    int c = vy * gx + vx;
                    for (int t = start[c]; t < start[c + 1]; t++) {
                        double dx = xi - pxy[2 * t], dy = yi - pxy[2 * t + 1];
                        double d2 = dx * dx + dy * dy;
                        if (d2 >= worst) continue;   /* reject without reading the id */
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
        int *out = w->nbr + (size_t)(i - 1) * K;
        for (int t = 0; t < K; t++) out[t] = (t < sz) ? hid[t] : 0;
    }
    #undef CELLX
    #undef CELLY
    #undef CLAMP
}

/* Build the kNN lists unless the requested K is already the one in memory.
   The savings and annealing phases may use different K; the flag is reset at
   the start of every instance. */
static void knn_need(const Inst *in, WS *w, int K)
{
    if (w->knn_k == K) return;
    ws_ensure(w, in->n, K, 1);
    knn_build(in, w, K);
    w->knn_k = K;
}

/* ------------------------------------------------------------ intra-route 2-opt
 * tour = 0, r[1..L], 0; improves to a local optimum (first improvement). */

static double two_opt(const Inst *in, int *t, int L)
{
    double gain = 0.0;
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int i = 1; i <= L - 1; i++) {
            int a = t[i - 1], b = t[i];
            double dab = dist(in, a, b);
            for (int j = i + 1; j <= L; j++) {
                int c = t[j], d = (j == L) ? 0 : t[j + 1];
                double delta = dist(in, a, c) + dist(in, b, d) - dab - dist(in, c, d);
                if (delta < -1e-10) {
                    for (int u = i, v = j; u < v; u++, v--) { int s = t[u]; t[u] = t[v]; t[v] = s; }
                    gain += delta; improved = 1;
                    a = t[i - 1]; b = t[i]; dab = dist(in, a, b);
                }
            }
        }
    }
    return gain;
}

/* =========================================================================
 *  Simulated annealing on the Clarke & Wright solution
 *
 *  Representation: doubly linked lists, one per route, closed on a virtual
 *  depot of their own (indices n+1 .. n+R, same coordinates as the real
 *  depot). An empty route is a virtual depot looping on itself: it stays
 *  available for a later reinsertion.
 *
 *  Operators (the second vertex is drawn among the nearest neighbours of the
 *  first, which avoids long-edge moves that would be rejected anyway):
 *    RELOCATE  move one customer elsewhere (intra or inter-route)   O(1)
 *    SWAP      exchange two customers                               O(1)
 *    2-OPT     same route      : segment reversal                   O(L)
 *              different routes: 2-opt* (tail exchange)             O(L)
 *
 *  Temperature: geometric decay from T0 to Tend over `steps` stages.
 *  The best solution seen is kept and restored at the end.
 * ========================================================================= */

typedef struct {
    int     n, R;
    int    *nxt, *prv, *rid;
    double *load;
} Sol;

#define VD(r)   (n + 1 + (r))               /* virtual depot of route r */
#define RE(v)   ((v) > n ? 0 : (v))         /* -> real index for dist()    */

/* Interleaved (x,y) coordinates per vertex, with the virtual depots
   materialised at indices n+1..: one cache line per vertex, and no more
   "is this a depot?" test inside the distance computation. */
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

/* materialise a route into an array; returns its length */
static int route_get(const Sol *S, int r, int *arr)
{
    const int n = S->n;
    int vd = VD(r), L = 0;
    for (int t = S->nxt[vd]; t != vd; t = S->nxt[t]) arr[L++] = t;
    return L;
}

/* rewrite route r from an array (updates rid and the load) */
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

static inline int sa_accept(double delta, double T, Rng *rng)
{
    if (delta <= 0.0) return 1;
    if (delta > 36.0 * T) return 0;    /* a multiply, avoiding the division */
    return rng_unit(rng) < exp(-delta / T);
}

/* lb[u] = sum of the two smallest distances that can be incident to u
   (among the depot and its k nearest neighbours). This is a lower bound on
   the cost of the two edges carried by u in any solution: the gap between
   the current cost and this bound measures how bad u's present neighbourhood
   is. */
static void lb_build(const Inst *in, WS *w, int K)
{
    const int n = in->n;
    for (int u = 1; u <= n; u++) {
        double a = dist(in, 0, u), b = 1e300;     /* two smallest */
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

/* ---------------------------------------------------------- Fenwick tree
 * Sampling exactly proportional to the weight p(u) = max(regret(u), 0) + eps.
 * fen[i] holds the sum of the weights over (i - lowbit(i), i]. The binary
 * descent through the tree gives an O(log n) draw and each weight update
 * costs O(log n); since only a few vertices change per accepted move (~2 % of
 * the steps), it is the sampling that dominates. */

static inline double fen_w(const WS *w, int u)
{
    double r = w->bad[u];
    return (r > 0.0 ? r : 0.0) + w->eps0;
}

static void fen_rebuild(WS *w, int n)
{
    double *ft = w->fen;
    for (int i = 1; i <= n; i++) ft[i] = fen_w(w, i);
    for (int i = 1; i <= n; i++) {
        int j = i + (i & -i);
        if (j <= n) ft[j] += ft[i];
    }
    double tot = 0.0;
    for (int i = 1; i <= n; i++) { w->fen_wv[i] = fen_w(w, i); tot += w->fen_wv[i]; }
    w->fen_tot = tot;
    int pw = 1; while ((pw << 1) <= n) pw <<= 1;
    w->fen_pw = pw;
}

static inline void fen_set(WS *w, int n, int u, double nv)
{
    double d = nv - w->fen_wv[u];
    if (d == 0.0) return;
    w->fen_wv[u] = nv; w->fen_tot += d;
    for (int i = u; i <= n; i += i & -i) w->fen[i] += d;
}

static inline int fen_sample(WS *w, int n, Rng *rng)
{
    double x = rng_unit(rng) * w->fen_tot;
    int pos = 0;
    for (int pw = w->fen_pw; pw > 0; pw >>= 1) {
        int np = pos + pw;
        if (np <= n && w->fen[np] <= x) { x -= w->fen[np]; pos = np; }
    }
    int u = pos + 1;
    return u > n ? n : u;
}

/* inc[u] = cost of the two edges currently carried by u. Maintained
   incrementally: only a few vertices change neighbourhood on each accepted
   move, and the acceptance rate is around 2 %, so the amortised cost is
   negligible next to recomputing it on every draw. */
static inline double bad_of(const Inst *in, const WS *w, const Sol *S, int u)
{
    (void)in;
    int p = S->prv[u], q = S->nxt[u];
    double c = w->inc[u];
    /* crit 0: gap to the theoretical bound lb[u] (two shortest possible
                edges) -- can stay large for an isolated customer even when
                nothing can improve it any further.
       crit 1: removal gain d(p,u)+d(u,q)-d(p,q), that is, exactly what a
                relocate of u would recover at best. Zero as soon as u sits
                well, and 2*d(0,u) if it is alone on its route. */
    if (w->crit == 3) ;                          /* raw cost of the 2 edges */
    else if (w->crit == 0) c -= w->lb[u];        /* gap to the bound        */
    else {
        c -= dxy(w, p, q);                       /* removal gain            */
        if (w->crit == 2) c /= w->lb[u];         /* ... normalised by the
                                                    local density           */
    }
    return c > 0.0 ? c : 0.0;
}

static inline void inc_upd(const Inst *in, WS *w, const Sol *S, int u)
{
    if (u < 1 || u > S->n) return;
    w->inc[u] = dxy(w, S->prv[u], u) + dxy(w, u, S->nxt[u]);
    if (w->pick_t != 1) w->bad[u] = bad_of(in, w, S, u);
    if (w->pick_t == 0) fen_set(w, S->n, u, fen_w(w, u));
}

static void inc_build(const Inst *in, WS *w, const Sol *S)
{
    const int n = S->n;
    for (int u = 1; u <= n; u++)
        w->inc[u] = dxy(w, S->prv[u], u) + dxy(w, u, S->nxt[u]);
    if (w->pick_t != 1)
        for (int u = 1; u <= n; u++) w->bad[u] = bad_of(in, w, S, u);
    if (w->pick_t == 0) fen_rebuild(w, n);
}

/* "regret" of u: cost of its two current edges minus the bound */
static inline double badness(const WS *w, int u) { return w->bad[u]; }

/* Choice of the vertex to move. Tournament of size t: draw t customers at
   random and keep the worst. t=1 gives back uniform sampling; the larger t,
   the higher the chance of drawing a badly placed vertex
   (P(rank r out of n) ~ t*(1-r/n)^(t-1)/n), with no structure to maintain. */
static inline int pick_u(const Inst *in, WS *w, const Sol *S, Rng *rng)
{
    (void)in;
    const uint64_t n = (uint64_t)S->n;
    const int t = w->pick_t;
    if (t == 0) return fen_sample(w, S->n, rng);
    if (t == 1) return 1 + rng_idx(rng, (int)n);

    int best = 1 + rng_idx(rng, (int)n);
    double bb = badness(w, best);
    for (int k = 1; k < t; k++) {
        int c = 1 + rng_idx(rng, (int)n);
        double bc = badness(w, c);
        if (bc > bb) { bb = bc; best = c; }
    }
    return best;
}

/* draw a candidate vertex "close" to u (or uniform if the list is empty) */
static inline int sa_cand(const WS *w, int n, int K, Rng *rng, int u)
{
    if (K <= 0) return 1 + rng_idx(rng, n);
    int v = w->nbr[(size_t)(u - 1) * K + rng_idx(rng, K)];
    return (v >= 1 && v <= n) ? v : 1 + rng_idx(rng, n);
}

/* ------------------------------------------------------------- RELOCATE */

static double mv_relocate(const Inst *in, WS *w, Sol *S, int K, double T, Rng *rng, int probe)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;

    int u = pick_u(in, w, S, rng);
    int v = sa_cand(w, n, K, rng, u);
    if (v == u) return 0.0;
    if (rng_bit(rng)) v = prv[v];              /* insert before v      */
    if (v == u || v == prv[u]) return 0.0;

    int p = prv[u], sc = nxt[u], q = nxt[v];
    int ru = rid[u], rv = rid[v];
    if (ru != rv && S->load[rv] + in->dem[u] > in->cap + EPS) return 0.0;

    /* d(p,u)+d(u,sc) is already held in inc[u]: 4 square roots instead of 6 */
    double delta = dxy(w, p, sc) - w->inc[u]
                 + dxy(w, v, u) + dxy(w, u, q) - dxy(w, v, q);
    if (probe) return delta;
    if (!sa_accept(delta, T, rng)) return 0.0;

    w->acc++;
    nxt[p] = sc; prv[sc] = p;
    nxt[v] = u;  prv[u]  = v;
    nxt[u] = q;  prv[q]  = u;
    S->load[ru] -= in->dem[u];
    S->load[rv] += in->dem[u];
    rid[u] = rv;
    {
        inc_upd(in, w, S, p);  inc_upd(in, w, S, sc); inc_upd(in, w, S, u);
        inc_upd(in, w, S, v);  inc_upd(in, w, S, q);
    }
    return delta;
}

/* ----------------------------------------------------------------- SWAP */

static double mv_swap(const Inst *in, WS *w, Sol *S, int K, double T, Rng *rng, int probe)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;

    int u = pick_u(in, w, S, rng);
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
        delta = dxy(w, pu, v) + dxy(w, u, sv)
              - dxy(w, pu, u) - dxy(w, v, sv);
        if (probe) return delta;
        if (!sa_accept(delta, T, rng)) return 0.0;
        w->acc++;
        nxt[pu] = v; prv[v] = pu;
        nxt[v] = u;  prv[u] = v;
        nxt[u] = sv; prv[sv] = u;
    } else if (sv == u) {                            /* v right before u */
        delta = dxy(w, pv, u) + dxy(w, v, su)
              - dxy(w, pv, v) - dxy(w, u, su);
        if (probe) return delta;
        if (!sa_accept(delta, T, rng)) return 0.0;
        w->acc++;
        nxt[pv] = u; prv[u] = pv;
        nxt[u] = v;  prv[v] = u;
        nxt[v] = su; prv[su] = v;
    } else {
        /* inc[u] = d(pu,u)+d(u,su) and inc[v] = d(pv,v)+d(v,sv): 4 instead of 8 */
        delta = dxy(w, pu, v) + dxy(w, v, su)
              + dxy(w, pv, u) + dxy(w, u, sv)
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
    {
        inc_upd(in, w, S, pu); inc_upd(in, w, S, su); inc_upd(in, w, S, u);
        inc_upd(in, w, S, pv); inc_upd(in, w, S, sv); inc_upd(in, w, S, v);
    }
    return delta;
}

/* -------------------------------------------------------------- OR-OPT
 * Moves a segment of 2 to `or_max` consecutive customers elsewhere (same
 * route or another), optionally reversing it. The edges internal to the
 * segment are unchanged, so the delta stays O(1); only detaching and
 * reattaching costs O(segment length), i.e. at most 3 links. */

static double mv_oropt(const Inst *in, WS *w, Sol *S, int K, double T, Rng *rng, int probe)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;

    int u = pick_u(in, w, S, rng);
    int want = 2 + rng_idx(rng, w->or_max - 1);
    int seg[8], L = 0;
    double sload = 0.0;
    for (int c = u; L < want && c >= 1 && c <= n; c = nxt[c]) {
        seg[L++] = c; sload += in->dem[c];
    }
    if (L < 2) return 0.0;                        /* route too short */

    int p = prv[seg[0]], q = nxt[seg[L - 1]];
    int ru = rid[seg[0]];

    int v = sa_cand(w, n, K, rng, u);
    if (rng_bit(rng)) v = prv[v];
    if (v == p) return 0.0;                       /* null move */
    for (int t = 0; t < L; t++) if (v == seg[t]) return 0.0;

    int rv = rid[v];
    if (ru != rv && S->load[rv] + sload > in->cap + EPS) return 0.0;

    int z = nxt[v];                               /* unchanged by the removal */
    int rev = rng_bit(rng);
    int first = rev ? seg[L - 1] : seg[0];
    int last  = rev ? seg[0] : seg[L - 1];

    double delta = dxy(w, p, q) - dxy(w, p, seg[0]) - dxy(w, seg[L - 1], q)
                 + dxy(w, v, first) + dxy(w, last, z) - dxy(w, v, z);
    if (probe) return delta;
    if (!sa_accept(delta, T, rng)) return 0.0;
    w->acc++;

    nxt[p] = q; prv[q] = p;                       /* detach */
    if (rev) for (int a = 0, b = L - 1; a < b; a++, b--) { int t = seg[a]; seg[a] = seg[b]; seg[b] = t; }
    int prevn = v;
    for (int t = 0; t < L; t++) {
        nxt[prevn] = seg[t]; prv[seg[t]] = prevn; rid[seg[t]] = rv; prevn = seg[t];
    }
    nxt[prevn] = z; prv[z] = prevn;
    S->load[ru] -= sload; S->load[rv] += sload;

    {
        inc_upd(in, w, S, p); inc_upd(in, w, S, q);
        inc_upd(in, w, S, v); inc_upd(in, w, S, z);
        for (int t = 0; t < L; t++) inc_upd(in, w, S, seg[t]);
    }
    return delta;
}

/* ---------------------------------------------------------------- 2-OPT */

static double mv_2opt(const Inst *in, WS *w, Sol *S, int K, double T, Rng *rng, int probe)
{
    const int n = S->n;
    int *nxt = S->nxt, *prv = S->prv, *rid = S->rid;

    int u = pick_u(in, w, S, rng);
    int v = sa_cand(w, n, K, rng, u);
    if (v == u) return 0.0;
    int a = rng_bit(rng) ? prv[u] : u;        /* edge (a, nxt[a]) */
    int b = rng_bit(rng) ? prv[v] : v;        /* edge (b, nxt[b]) */
    if (a == b) return 0.0;

    int ra = rid[a], rb = rid[b];

    if (ra == rb) {                                  /* ---- intra 2-opt ---- */
        int sa = nxt[a], sb = nxt[b];
        if (sa == b || sb == a) return 0.0;          /* empty segment */
        double delta = dxy(w, a, b) + dxy(w, sa, sb)
                     - dxy(w, a, sa) - dxy(w, b, sb);
        if (probe) return delta;
        if (!sa_accept(delta, T, rng)) return 0.0;
        w->acc++;

        /* On a cycle, reversing the segment nxt[a]..b or its complement
           yields the same undirected cycle: the relative order of a and b
           therefore need not be known (this saves an O(L) walk on *every*
           draw), and the segment may contain the virtual depot harmlessly.
           Only the reversal, which happens on acceptance, is O(L). */
        int *buf = w->bufA, L = 0;
        for (int t = sa; ; t = nxt[t]) { buf[L++] = t; if (t == b) break; }
        int p = a;
        for (int t = L - 1; t >= 0; t--) { nxt[p] = buf[t]; prv[buf[t]] = p; p = buf[t]; }
        nxt[p] = sb; prv[sb] = p;
        {
            inc_upd(in, w, S, a);  inc_upd(in, w, S, sa);
            inc_upd(in, w, S, b);  inc_upd(in, w, S, sb);
        }
        return delta;
    }

    /* ---- 2-opt*: exchange the tails of the two routes ---- */
    int sa = nxt[a], sb = nxt[b];
    double delta = dxy(w, a, sb) + dxy(w, b, sa)
                 - dxy(w, a, sa) - dxy(w, b, sb);
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
    {
        inc_upd(in, w, S, a);  inc_upd(in, w, S, sa);
        inc_upd(in, w, S, b);  inc_upd(in, w, S, sb);
    }
    return delta;
}

/* =========================================================================
 *  Optimal Split in O(n)  (Vidal, "Technical note: Split algorithm in O(n)
 *  for the capacitated vehicle routing problem", C&OR 2016; Prins 2004
 *  formulation for the auxiliary graph)
 *
 *  The current routes are concatenated into a giant tour t_1..t_n, which is
 *  then optimally repartitioned. Since the current partition is itself a
 *  feasible cut of that tour, the cost cannot increase.
 *
 *  With D_i = sum of the demands up to t_i and C_i = length of the path
 *  t_1..t_i, a route (i, j] costs  d0(t_{i+1}) + (C_j - C_{i+1}) + d0(t_j),
 *  hence the recurrence
 *        P_j = C_j + d0(t_j) + min { F(i) : D_j - D_i <= Q },
 *        F(i) = P_i + d0(t_{i+1}) - C_{i+1}.
 *  F does not depend on j and the feasible window is a sliding interval
 *  (D is increasing): a monotone-deque sliding minimum gives the O(n).
 * ========================================================================= */

typedef struct { double ang; int r; } RAng;

static int cmp_rang(const void *a, const void *b)
{
    double x = ((const RAng *)a)->ang, y = ((const RAng *)b)->ang;
    return (x > y) - (x < y);
}

/* Builds the giant tour. In "sweep" mode the routes are first sorted by the
   polar angle of their centroid around the depot, then each route is oriented
   so as to minimise the join with the previous one: the concatenation becomes
   a plausible tour, and Split can then actually move the cuts. In "routes"
   mode we concatenate in the current order -- the optimal cuts are then
   almost always the starting ones. */
static double split_apply_mode(const Inst *in, WS *w, Sol *S, int sweep)
{
    const int n = S->n;
    int *t = w->tour;                            /* giant tour, 1..n */
    int L = 0;

    if (!sweep) {
        for (int r = 0; r < S->R; r++) {
            int vd = VD(r);
            for (int c = S->nxt[vd]; c != vd; c = S->nxt[c]) t[++L] = c;
        }
    } else {
        RAng *ra = (RAng *)w->spAng;
        int nr = 0;
        for (int r = 0; r < S->R; r++) {
            int vd = VD(r), cnt = 0;
            double sx = 0, sy = 0;
            for (int c = S->nxt[vd]; c != vd; c = S->nxt[c]) {
                sx += in->x[c]; sy += in->y[c]; cnt++;
            }
            if (!cnt) continue;
            ra[nr].ang = atan2(sy / cnt - in->y[0], sx / cnt - in->x[0]);
            ra[nr].r = r; nr++;
        }
        qsort(ra, (size_t)nr, sizeof(RAng), cmp_rang);
        int last = 0;                            /* depot */
        for (int k = 0; k < nr; k++) {
            int vd = VD(ra[k].r);
            int head = S->nxt[vd], tailc = S->prv[vd];
            int rev = dist(in, RE(last), RE(tailc)) < dist(in, RE(last), RE(head));
            if (!rev) for (int c = head;  c != vd; c = S->nxt[c]) t[++L] = c;
            else      for (int c = tailc; c != vd; c = S->prv[c]) t[++L] = c;
            last = t[L];
        }
    }
    if (L != n) return -1.0;                     /* inconsistent solution */

    double *D = w->spD, *C = w->spC, *P = w->spP, *F = w->spF;
    int *pred = w->spPred, *dq = w->spDq;

    const double Qcap = in->cap + EPS;
    D[0] = 0.0; C[1] = 0.0;
    for (int i = 1; i <= n; i++) {
        /* The sliding window assumes a single customer fits in a vehicle:
           without that, no feasible predecessor exists for this customer, the
           deque empties and we read out of bounds. On an infeasible instance
           we simply give up on Split. */
        if (in->dem[t[i]] > Qcap) return -1.0;
        D[i] = D[i - 1] + in->dem[t[i]];
    }
    for (int i = 2; i <= n; i++) C[i] = C[i - 1] + dist(in, t[i - 1], t[i]);

    const double Q = Qcap;
    int head = 0, tail = 0;                      /* deque dq[head..tail-1] */
    P[0] = 0.0;
    F[0] = P[0] + dist(in, 0, t[1]) - C[1];
    dq[tail++] = 0;

    for (int j = 1; j <= n; j++) {
        if (j >= 2) {                            /* insert candidate j-1 */
            while (tail > head && F[dq[tail - 1]] >= F[j - 1]) tail--;
            dq[tail++] = j - 1;
        }
        while (head < tail - 1 && D[j] - D[dq[head]] > Q) head++;  /* capacity */
        if (D[j] - D[dq[head]] > Q) return -1.0;   /* guard: cannot happen
                                        after the test above */
        int i = dq[head];
        P[j] = F[i] + C[j] + dist(in, 0, t[j]);
        pred[j] = i;
        if (j < n) F[j] = P[j] + dist(in, 0, t[j + 1]) - C[j + 1];
    }

    /* --- reconstruction --- */
    int nr = 0;
    for (int j = n; j > 0; j = pred[j]) nr++;
    if (nr > n) return -1.0;

    int *cut = w->spDq;                          /* reused after the deque */
    int k = nr;
    for (int j = n; j > 0; j = pred[j]) cut[--k] = j;   /* route end points */

    int start = 0;
    for (int r = 0; r < nr; r++) {
        route_set(S, in, r, t + start + 1, cut[r] - start);
        start = cut[r];
    }
    for (int r = nr; r < S->R; r++) {            /* routes now empty */
        int vd = VD(r);
        S->nxt[vd] = vd; S->prv[vd] = vd; S->rid[vd] = r; S->load[r] = 0.0;
    }
    if (nr > S->R) S->R = nr;
    return P[n];
}

/* Applies Split under both constructions and keeps the better one. */
static double split_apply(const Inst *in, WS *w, const Opts *o, Sol *S, double cur)
{
    double best = cur;
    if (o->split_tour != 1) {                    /* direct concatenation */
        double p = split_apply_mode(in, w, S, 0);
        if (p >= 0.0 && p < best - 1e-12) best = p;
        else if (p < 0.0) return -1.0;
    }
    if (o->split_tour != 0) {                    /* angular sweep          */
        double p = split_apply_mode(in, w, S, 1);
        if (p < 0.0) return -1.0;
        if (p < best - 1e-12) best = p;
        else if (p > best + 1e-12) {             /* sweep made it worse:   */
            split_apply_mode(in, w, S, 0);       /* fall back to the       */
        }                                        /* current order (idempotent) */
    }
    return best;
}

/* --------------------------------------------------------- drawing a move */

static inline double sa_draw(const Inst *in, WS *w, const Opts *o, Sol *S,
                             int K, double T, Rng *rng, int probe)
{
    uint32_t z = rng32(rng);
    if (z < w->th1) return mv_relocate(in, w, S, K, T, rng, probe);
    if (z < w->th2) return mv_swap    (in, w, S, K, T, rng, probe);
    if (z < w->th3) return mv_2opt    (in, w, S, K, T, rng, probe);
    return                 mv_oropt   (in, w, S, K, T, rng, probe);
}

/* ------------------------------------------- calibration of the initial T
 * We draw moves in probe mode from the starting solution and keep the mean
 * of the worsening deltas Delta+; the initial temperature is then
 *      T0 = -mean(Delta+) / ln(chi0)
 * where chi0 is the proportion of worsening moves we want to accept at the
 * start (Ben-Ameur, 2004). The final temperature follows `decades` decades
 * below it. */

static double calibrate_T0(const Inst *in, WS *w, const Opts *o, Sol *S,
                           int K, Rng *rng)
{
    const int maxdraw = 2000, want = 300;
    double sum = 0.0;
    int ns = 0;
    for (int t = 0; t < maxdraw && ns < want; t++) {
        double d = sa_draw(in, w, o, S, K, 1.0, rng, 1);
        if (d > 0.0) { sum += d; ns++; }
    }
    if (!ns) return 0.0;                       /* no worsening move at all */
    return -(sum / ns) / log(o->sa_chi0);
}

/* ------------------------------------------------------- the annealing loop */

static double anneal(const Inst *in, WS *w, const Opts *o, Sol *S,
                     double cur, uint64_t seed)
{
    const int n = S->n, R = S->R, steps = o->sa_steps;
    if (steps <= 0) return cur;

    int K = o->sa_knn; if (K > n - 1) K = n - 1; if (K < 0) K = 0;
    w->pick_t = (K > 0) ? o->pick_t : 1;   /* the regret needs the kNN lists */
    w->crit = o->pick_crit; w->or_max = o->or_max;
    {   double wt = o->w_rel + o->w_swap + o->w_2opt + o->w_or;
        double c1 = o->w_rel / wt, c2 = c1 + o->w_swap / wt, c3 = c2 + o->w_2opt / wt;
        w->th1 = (uint32_t)(c1 * 4294967296.0);
        w->th2 = (uint32_t)(c2 * 4294967296.0);
        w->th3 = (uint32_t)(c3 * 4294967296.0);
        if (o->w_or <= 0.0) w->th3 = 0xFFFFFFFFu;
    }
    xy_build(in, w);
    if (K > 0) {
        knn_need(in, w, K); lb_build(in, w, K);
        double m = 0.0;
        for (int u = 1; u <= n; u++) m += w->lb[u];
        w->eps0 = o->pick_eps * m / (2.0 * n);   /* ~ nearest-neighbour distance */
    } else if (w->crit == 0 || w->crit == 2) w->pick_t = 1;  /* these need the kNN */
    else w->eps0 = o->pick_eps * cur / (double)(n + R);
    inc_build(in, w, S);

    Rng rng; rng_seed(&rng, seed);
    const size_t zi = (size_t)(2 * n + 4) * sizeof(int);
    const size_t zd = (size_t)(n + 2) * sizeof(double);
    int bR = R;

    /* temperatures: calibrated by sampling, unless --t0 is forced */
    double T0 = o->sa_t0, Tend = o->sa_tend;
    if (T0 <= 0.0) {                                   /* automatic mode */
        T0 = calibrate_T0(in, w, o, S, K, &rng);
        if (T0 <= 0.0) return cur;                     /* frozen solution   */
        Tend = T0 * pow(10.0, -o->sa_decades);
    }
    double T = T0;
    const double alpha = (steps > 1) ? pow(Tend / T0, 1.0 / (steps - 1)) : 1.0;
    double best = cur;
    memcpy(w->b_nxt, S->nxt, zi); memcpy(w->b_prv, S->prv, zi);
    memcpy(w->b_rid, S->rid, zi); memcpy(w->b_load, S->load, zd);

    for (int it = 0; it < steps; it++) {
        cur += sa_draw(in, w, o, S, K, T, &rng, 0);

        if (o->split_every > 0 && (it + 1) % o->split_every == 0) {
            double p = split_apply(in, w, o, S, cur);
            if (p >= 0.0) { w->gsplit += cur - p; cur = p;
                            inc_build(in, w, S); }
        }
        if (cur < best - 1e-12) {
            best = cur;
            bR = S->R;
            memcpy(w->b_nxt, S->nxt, zi); memcpy(w->b_prv, S->prv, zi);
            memcpy(w->b_rid, S->rid, zi); memcpy(w->b_load, S->load, zd);
        }
        T *= alpha;
    }
    memcpy(S->nxt, w->b_nxt, zi); memcpy(S->prv, w->b_prv, zi);
    memcpy(S->rid, w->b_rid, zi); memcpy(S->load, w->b_load, zd);
    S->R = bR;
    inc_build(in, w, S);
    w->t0_used = T0;
    return best;
}

/* =========================================================================
 *  Core: Clarke & Wright
 * ========================================================================= */

typedef struct {
    double cost, cost0, drift, acc, t0, gsplit, mean0, mean1, worst1;
    int    routes;
    double load_max;
    int    feasible;
    double time_ms;
} Result;

static double solve_cw(const Inst *in, WS *w, const Opts *o,
                       Result *res, int **sol_out, uint64_t seed)
{
    const int n = in->n;
    if ((unsigned)n > MAXN) die("n=%d exceeds %u", n, MAXN);
    w->knn_k = 0;                                /* kNN lists must be rebuilt */

    /* --- size of the savings list --- */
    int K = o->knn;
    int exact;
    if (K < 0)       { exact = 1; }                            /* --exact */
    else if (K == 0) { exact = (n <= 1500); K = 32; }          /* auto    */
    else             { exact = 0; }
    if (exact) K = n - 1;
    if (K > n - 1) K = n - 1;
    if (K < 1) K = 1;

    size_t nsav = exact ? ((size_t)n * (size_t)(n - 1)) / 2 : (size_t)n * (size_t)K;
    if (o->init_random) nsav = 1;         /* no savings list is ever built */
    ws_ensure(w, n, K, nsav ? nsav : 1);

    double *d0 = w->d0;
    for (int i = 0; i <= n; i++) d0[i] = dist(in, 0, i);

    Sol sol;
    sol.n = n;
    sol.nxt = w->s_nxt; sol.prv = w->s_prv; sol.rid = w->s_rid; sol.load = w->s_load;

    const size_t zi = (size_t)(2 * n + 4) * sizeof(int);
    const size_t zd = (size_t)(n + 2) * sizeof(double);
    double bestc = 1e300, bestc0 = 0, sum0 = 0, sum1 = 0, worst1 = -1e300, gsplit_tot = 0;
    int bestR = 0, best_visited = 0;
    Result rbest; memset(&rbest, 0, sizeof rbest);

    /* ======================= multiple restarts ======================= */
    for (int rs = 0; rs < o->restarts; rs++) {

    Rng rcw; rng_seed(&rcw, seed * 0xD1342543DE82EF95ULL + (uint64_t)rs * 0x9E3779B97F4A7C15ULL);
    const int rnd = (rs > 0) ? o->cw_rand : 0;      /* restart 0 = deterministic C&W */
    /* multiplicative perturbation of the savings: changes the order in which
       the pairs (hence the customers) are considered by the algorithm */
    const double alpha = (rnd & 1) ? o->cw_alpha : 0.0;
    /* jitter on Yellow's parameters */
    double lam = o->lambda, mu = o->mu;
    if (rnd & 2) { lam *= 0.75 + 0.5 * rng_unit(&rcw); mu += 0.3 * rng_unit(&rcw); }

    const double cap = in->cap;
    int *flat = w->flat, *rstart = w->rstart, *rt = w->route;
    int *seen = w->seen;
    int R = 0, nf = 0, visited = 0;
    for (int i = 1; i <= n; i++) seen[i] = 0;

    if (o->init_random) {
    /* ---------------------------------------------- random feasible start
     * Shuffle the customers and cut the sequence whenever the next one would
     * overflow the vehicle. Feasible by construction, which matters: the
     * annealing operators reject any capacity violation, so an infeasible
     * start could never be repaired. First-fit on a random permutation also
     * lands close to the savings construction's route count, so the two
     * starts differ in quality rather than in shape.
     *
     * Every restart draws its own permutation from rcw, so --restarts gives a
     * genuine multi-start. --cw-rand, --cw-alpha, --lambda, --mu and the
     * savings list size have no effect on this path. */
    int *perm = w->bufA;
    for (int i = 0; i < n; i++) perm[i] = i + 1;
    for (int i = n - 1; i > 0; i--) {                /* Fisher-Yates */
        int j = rng_idx(&rcw, i + 1);
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    int start = 0;
    double load = 0.0;
    for (int i = 0; i <= n; i++) {
        int flush = (i == n) ||
                    (i > start && load + in->dem[perm[i]] > cap + EPS);
        if (flush && i > start) {
            int L = i - start;
            for (int t = 0; t < L; t++) rt[t + 1] = perm[start + t];
            if (o->do2opt && L >= 3) { rt[0] = 0; two_opt(in, rt, L); }
            rstart[R++] = nf;
            for (int t = 1; t <= L; t++) { flat[nf++] = rt[t]; seen[rt[t]] = 1; }
            visited += L;
            start = i; load = 0.0;
        }
        if (i < n) load += in->dem[perm[i]];
    }

    } else {
        /* --- savings generation --- */
        Sav *S = w->sav;
        size_t m = 0;

        if (exact) {
            for (int i = 1; i <= n; i++) {
                double d0i = d0[i];
                for (int j = i + 1; j <= n; j++) {
                    double s = d0i + d0[j] - lam * dist(in, i, j) + mu * fabs(d0i - d0[j]);
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
                    double s = d0i + d0[j] - lam * dist(in, i, j) + mu * fabs(d0i - d0[j]);
                    if (alpha > 0.0) s *= 1.0 + alpha * (2.0 * rng_unit(&rcw) - 1.0);
                    if (s <= 0.0) continue;
                    float sf = (float)s; uint32_t k;
                    memcpy(&k, &sf, 4);
                    S[m].key = k; S[m].ij = ((uint32_t)a << 16) | (uint32_t)b; m++;
                }
            }
        }

        radix_sort(S, w->tmp, m);

        /* --- route merging (decreasing savings) --- */
        int *uf = w->uf, *deg = w->deg, *adj = w->adj;
        double *load = w->load;
        for (int i = 1; i <= n; i++) { uf[i] = i; deg[i] = 0; load[i] = in->dem[i]; }

        for (size_t t = m; t-- > 0; ) {
            uint32_t ij = S[t].ij;
            int i = (int)(ij >> 16), j = (int)(ij & 0xFFFF);
            if (deg[i] >= 2 || deg[j] >= 2) continue;    /* a route endpoint? */
            int ri = uf_find(uf, i), rj = uf_find(uf, j);
            if (ri == rj) continue;                      /* already same route   */
            if (load[ri] + load[rj] > cap + EPS) continue;
            adj[2 * i + deg[i]++] = j;
            adj[2 * j + deg[j]++] = i;
            uf[rj] = ri; load[ri] += load[rj];
        }

        for (int s = 1; s <= n; s++) {
            if (seen[s] || deg[s] >= 2) continue;        /* start from an endpoint */
            int L = 0, prev = 0, cur = s;
            while (cur) {
                rt[++L] = cur; seen[cur] = 1; visited++;
                int nx = 0;
                for (int e = 0; e < deg[cur]; e++)
                    if (adj[2 * cur + e] != prev) { nx = adj[2 * cur + e]; break; }
                prev = cur; cur = nx;
            }
            if (o->do2opt && L >= 3) { rt[0] = 0; two_opt(in, rt, L); }
            rstart[R++] = nf;
            for (int t = 1; t <= L; t++) flat[nf++] = rt[t];
        }
    }
    rstart[R] = nf;

    /* --- linked solution, cost before annealing --- */
    sol.R = R;
    for (int r = 0; r < R; r++)
        route_set(&sol, in, r, flat + rstart[r], rstart[r + 1] - rstart[r]);

    w->gsplit = 0.0;
    double cost0 = sol_cost(in, &sol);
    if (o->split & 1) {                          /* Split right after C&W */
        double p = split_apply(in, w, o, &sol, cost0);
        if (p >= 0.0) { w->gsplit += cost0 - p; cost0 = p; }
    }
    double cost = cost0;

    /* --- simulated annealing --- */
    if (o->sa_steps > 0) {
        w->acc = 0;
        double tracked = anneal(in, w, o, &sol, cost0,
                                (seed ^ 0x5DEECE66DULL) + (uint64_t)rs * 0xBF58476D1CE4E5B9ULL);
        cost = sol_cost(in, &sol);
        res->drift = fabs(cost - tracked);
        res->acc = (double)w->acc / o->sa_steps;
        res->t0  = w->t0_used;
    } else {
        res->drift = 0.0; res->acc = 0.0; res->t0 = 0.0;
    }
    if (o->split & 2) {                          /* final Split */
        double p = split_apply(in, w, o, &sol, cost);
        if (p >= 0.0) { w->gsplit += cost - p; cost = sol_cost(in, &sol); }
    }

    sum0 += cost0; sum1 += cost; gsplit_tot += w->gsplit;
    if (cost > worst1) worst1 = cost;
    if (cost < bestc) {                          /* best restart so far */
        bestc = cost; bestc0 = cost0; bestR = sol.R; best_visited = visited;
        rbest = *res;
        memcpy(w->r_nxt, sol.nxt, zi); memcpy(w->r_prv, sol.prv, zi);
        memcpy(w->r_rid, sol.rid, zi); memcpy(w->r_load, sol.load, zd);
    }
    }   /* ==================== end of multiple restarts ==================== */

    memcpy(sol.nxt, w->r_nxt, zi); memcpy(sol.prv, w->r_prv, zi);
    memcpy(sol.rid, w->r_rid, zi); memcpy(sol.load, w->r_load, zd);
    sol.R = bestR;
    int R = bestR, visited = best_visited;
    (void)R;
    *res = rbest;
    double cost = bestc, cost0 = bestc0;
    res->gsplit = gsplit_tot / o->restarts;
    res->mean0  = sum0 / o->restarts;
    res->mean1  = sum1 / o->restarts;
    res->worst1 = worst1;

    /* --- checking and output --- */
    int *seen = w->seen, *rt = w->route;
    int *ob = NULL, obn = 1;
    if (sol_out) ob = (int *)xmalloc((size_t)(n + bestR + 3) * sizeof(int));
    double lmax = 0.0;
    int nroutes = 0, feasible = (visited == n);
    for (int i = 1; i <= n; i++) seen[i] = 0;
    for (int r = 0; r < R; r++) {
        int vd = n + 1 + r, L = 0;
        double rload = 0.0;
        for (int t = sol.nxt[vd]; t != vd; t = sol.nxt[t]) {
            if (t < 1 || t > n || seen[t]) { feasible = 0; break; }
            seen[t] = 1; rload += in->dem[t]; rt[++L] = t;
        }
        if (!L) continue;                            /* route emptied by annealing */
        nroutes++;
        if (rload > lmax) lmax = rload;
        if (rload > in->cap + EPS) feasible = 0;
        if (ob) {                                /* [n_words, c.., 0, c.., 0] */
            for (int t = 1; t <= L; t++) ob[obn++] = rt[t];
            ob[obn++] = 0;
        }
    }
    for (int i = 1; i <= n; i++) if (!seen[i]) feasible = 0;
    if (ob) { ob[0] = obn - 1; *sol_out = ob; }
    res->cost0 = cost0;

    res->cost = cost; res->routes = nroutes;
    res->load_max = lmax; res->feasible = feasible;
    return cost;
}

/* ------------------------------------------------------------ .cvrpb export */

static void write_bundle(const char *path, const Inst *v, int M)
{
    FILE *f = fopen(path, "wb");
    if (!f) die("writing %s: %s", path, strerror(errno));
    uint32_t nb = (uint32_t)M, rsv = 0;
    fwrite("CVRPBIN1", 1, 8, f);
    fwrite(&nb, 4, 1, f); fwrite(&rsv, 4, 1, f);
    for (int k = 0; k < M; k++) {
        uint32_t n = (uint32_t)v[k].n;
        size_t sz = (size_t)n + 1;
        fwrite(&n, 4, 1, f); fwrite(&v[k].cap, 8, 1, f);
        fwrite(v[k].x, 8, sz, f); fwrite(v[k].y, 8, sz, f); fwrite(v[k].dem, 8, sz, f);
    }
    fclose(f);
}

/* --------------------------------------------------------------- validation
 * Re-reads a solution file and confronts it with the instances: every
 * customer served exactly once, capacities respected, and the cost recomputed
 * from the coordinates then compared with the one written in the file. No
 * solver state is reused: only the instances and the file are read. */

static int validate_file(const char *path, const Inst *insts, int M, const Opts *o)
{
    FILE *f = fopen(path, "r");
    if (!f) die("reading %s: %s", path, strerror(errno));

    char line[8192];
    int errors = 0, checked = 0, file_round = -1;
    double worst_cost = 0.0;
    Inst tmp; memset(&tmp, 0, sizeof tmp);
    int tmp_alloc = 0;
    int *seen = NULL; int seen_n = 0;

    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') { sscanf(line, "#round %d", &file_round); continue; }
        if (line[0] == '\n') continue;
        int idx, n, nr; char name[128]; double cap, cost;
        if (sscanf(line, "inst %d %127s %d %lf %lf %d",
                   &idx, name, &n, &cap, &cost, &nr) != 6) continue;
        if (idx < 0 || idx >= M) { fprintf(stderr, "index %d out of range\n", idx); errors++; continue; }

        if (file_round >= 0 && file_round != (o->rounded ? 1 : 0)) {
            fprintf(stderr, "warning: this file was produced with --round %d,"
                    " re-run the validation with the same option\n", file_round);
            errors++; break;
        }
        const Inst *in;
        if (o->random) {
            if (!tmp_alloc || tmp.n < n) {
                if (tmp_alloc) inst_free(&tmp);
                inst_alloc(&tmp, n); tmp_alloc = 1;
            }
            gen_instance(&tmp, n, cap, o->seed + (uint64_t)idx, o->rounded);
            in = &tmp;
        } else in = &insts[idx];

        if (in->n != n) { fprintf(stderr, "inst %d: n=%d, expected %d\n", idx, n, in->n); errors++; }
        if (n + 1 > seen_n) { seen = (int *)xrealloc(seen, (size_t)(n + 1) * sizeof(int)); seen_n = n + 1; }
        for (int i = 1; i <= n; i++) seen[i] = 0;

        double total = 0.0;
        int served = 0, bad = 0;
        for (int r = 0; r < nr; r++) {
            if (!fgets(line, sizeof line, f)) { fprintf(stderr, "inst %d: truncated routes\n", idx); errors++; bad = 1; break; }
            double load = 0.0, rc = 0.0;
            int prev = 0, cnt = 0;
            for (char *p = line; ; ) {
                while (*p == ' ' || *p == '\t') p++;
                if (*p < '0' || *p > '9') break;
                int c = (int)strtol(p, &p, 10);
                if (c < 1 || c > n) { fprintf(stderr, "inst %d: invalid customer %d\n", idx, c); errors++; bad = 1; break; }
                if (seen[c]) { fprintf(stderr, "inst %d: customer %d served twice\n", idx, c); errors++; bad = 1; }
                seen[c] = 1; served++; cnt++;
                rc += dist(in, prev, c); load += in->dem[c]; prev = c;
            }
            if (bad) break;
            if (!cnt) continue;
            rc += dist(in, prev, 0);
            total += rc;
            if (load > in->cap + EPS) {
                fprintf(stderr, "inst %d: route %d overloaded (%g > %g)\n", idx, r, load, in->cap);
                errors++;
            }
        }
        if (bad) continue;
        if (served != n) { fprintf(stderr, "inst %d: %d customers served out of %d\n", idx, served, n); errors++; }
        double rel = fabs(total - cost) / (cost > 0 ? cost : 1.0);
        if (rel > worst_cost) worst_cost = rel;
        if (rel > 1e-9) {
            fprintf(stderr, "inst %d: reported cost %.10f, recomputed %.10f (gap %.2e)\n",
                    idx, cost, total, rel);
            errors++;
        }
        checked++;
    }
    fclose(f);
    if (tmp_alloc) inst_free(&tmp);
    free(seen);
    printf("validation of %s: %d instances checked, %d error(s)\n", path, checked, errors);
    printf("  max relative gap between reported and recomputed cost: %.3e\n", worst_cost);
    return errors;
}

/* ============================================================== main / CLI */

static void usage(void)
{
    printf(
"Clarke & Wright (savings) for the CVRP — parallel C implementation\n"
"\n"
"Usage: cw [source] [options]\n"
"\n"
"Source (exactly one):\n"
"  --dir DIR          every .vrp/.txt/.cvrpb instance in the directory\n"
"  --bundle FILE      one .cvrpb file (bundle of instances)\n"
"  --random           randomly generated instances (Kool/NeuOpt law:\n"
"                     coordinates U[0,1]^2, demands U{1..9})\n"
"\n"
"Generation options (--random):\n"
"  -n N               number of customers          (default 100)\n"
"  -m M               number of instances          (default 1000)\n"
"  --cap C            capacity (default: 30/40/50/70 for n=20/50/100/200)\n"
"  --seed S           random seed                  (default 42)\n"
"\n"
"Initial solution:\n"
"  --init cw|random   cw (default) = Clarke & Wright savings construction;\n"
"                     random = shuffle the customers and cut into routes at\n"
"                     the capacity limit. With --restarts each restart draws\n"
"                     its own permutation. The savings options below have no\n"
"                     effect on the random path.\n"
"\n"
"Heuristic options:\n"
"  --knn K            savings limited to the K nearest neighbours\n"
"                     0 = automatic (exact if n<=1500), default 0\n"
"  --exact            full savings list (O(n^2))\n"
"  --lambda L         savings = d0i+d0j - L*dij + mu*|d0i-d0j|  (default 1)\n"
"  --mu M             (default 0)\n"
"  --2opt             intra-route 2-opt after construction\n"
"  --round            integer distances (TSPLIB EUC_2D convention);\n"
"                     applies to every instance source\n"
"\n"
"Simulated annealing (on by default, applied after Clarke & Wright):\n"
"  --sa-steps N       number of steps              (default 1000)\n"
"  --no-sa            disable annealing\n"
"  --ops r,s,t,o      relative weights of the relocate, swap, 2-opt and\n"
"                     or-opt operators  (default 1,1,1,0: or-opt disabled,\n"
"                     see the README; e.g. --ops 1,0,0,0 for relocate only)\n"
"  --or-max L         maximum or-opt segment length              (default 3)\n"
"  --t-accept X       target initial acceptance rate         (default 0.001)\n"
"                     T0 is calibrated by sampling the worsening moves\n"
"  --t-decades D      number of decades spanned by T            (default 2)\n"
"  --t0 T             set T0 by hand (disables calibration)\n"
"  --tend T           set the final temperature       (default T0 * 1e-4)\n"
"  --sa-knn K         candidate neighbourhood size (default 20, 0 = uniform)\n"
"  --pick T           choice of the vertex to move: tournament of size T on\n"
"                     the regret (cost of the two edges carried by the vertex\n"
"                     minus the bound of the two shortest possible ones).\n"
"                     T=1: uniform sampling, T=0: sampling exactly\n"
"                     proportional to the regret via a Fenwick tree.\n"
"                                                              (default 2)\n"
"  --pick-crit C      definition of the regret: rem = removal gain\n"
"                     d(p,u)+d(u,q)-d(p,q), attainable; lb = gap to the bound\n"
"                     of the two shortest edges; remnorm = removal gain\n"
"                     divided by that bound (normalised by the local\n"
"                     density); raw = raw cost of the two edges (default lb)\n"
"  --pick-eps E       with --pick 0: weight = regret + E * (mean distance to\n"
"                     the nearest neighbour); small E = more peaked sampling\n"
"                     (default 0.3)\n"
"  --check            report the incremental-cost drift (verification)\n"
"\n"
"Multiple restarts:\n"
"  --restarts R       R initial C&W solutions per instance; annealing is run\n"
"                     from each of them and the best one is kept\n"
"                     (restart 0 is always the deterministic C&W)\n"
"  --cw-rand MODE     off | perturb | param | both       (default perturb)\n"
"                     perturb: each saving is multiplied by 1+alpha*U(-1,1),\n"
"                              which reshuffles the order in which pairs of\n"
"                              customers are considered\n"
"                     param  : lambda and mu are drawn (Yellow savings)\n"
"  --cw-alpha A       perturbation amplitude                 (default 0.03)\n"
"\n"
"Optimal Split (Vidal 2016, O(n)): repartitions the giant tour obtained by\n"
"concatenating the routes; can never make the cost worse.\n"
"  --split MODE       off | cw | end | both             (default off)\n"
"  --split-every N    also apply Split every N annealing steps\n"
"  --split-tour T     construction of the giant tour:\n"
"                     routes = concatenation in the current order\n"
"                     sweep  = routes sorted by polar angle and oriented\n"
"                     both   = both, keeping the better one    (default)\n"
"\n"
"Miscellaneous:\n"
"  --threads T        number of OpenMP threads (default: all)\n"
"  --limit L          process only the first L instances\n"
"  --csv FILE         write a per-instance CSV\n"
"  --sol FILE         write every solution (self-describing text format,\n"
"                     identical whatever the thread count)\n"
"  --dump-bundle F    write the instances used, in .cvrpb format\n"
"  --validate F       solves nothing: re-reads a solution file, checks\n"
"                     coverage and capacities, recomputes each cost from the\n"
"                     coordinates and compares it with the reported value\n"
"  --per-instance     print the cost of every instance\n"
"  -q                 quiet\n"
"  -h, --help         this help\n");
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
    o.n = 100; o.m = 1000; o.seed = 42; o.cap = -1;
    o.lambda = 1.0; o.mu = 0.0; o.threads = 0; o.knn = 0;
    o.sa_steps = 1000; o.sa_t0 = -1.0; o.sa_tend = -1.0; o.sa_chi0 = 0.001; o.sa_decades = 2.0; o.split = 0; o.split_every = 0; o.split_tour = 2;
    o.restarts = 1; o.cw_rand = 1; o.cw_alpha = 0.03; o.pick_t = 2; o.pick_eps = 0.3; o.pick_crit = 0; o.sa_knn = 20;
    o.w_rel = 1.0; o.w_swap = 1.0; o.w_2opt = 1.0; o.w_or = 0.0; o.or_max = 3;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT() (i + 1 < argc ? argv[++i] : (die("option %s: missing argument", a), ""))
        if      (!strcmp(a, "--dir"))          o.dir = NEXT();
        else if (!strcmp(a, "--bundle"))       o.bundle = NEXT();
        else if (!strcmp(a, "--random"))       o.random = 1;
        else if (!strcmp(a, "-n"))             o.n = atoi(NEXT());
        else if (!strcmp(a, "-m"))             o.m = atoi(NEXT());
        else if (!strcmp(a, "--cap"))          o.cap = atof(NEXT());
        else if (!strcmp(a, "--seed"))         o.seed = strtoull(NEXT(), NULL, 10);
        else if (!strcmp(a, "--knn"))          o.knn = atoi(NEXT());
        else if (!strcmp(a, "--exact"))        o.knn = -1;
        else if (!strcmp(a, "--lambda"))       o.lambda = atof(NEXT());
        else if (!strcmp(a, "--mu"))           o.mu = atof(NEXT());
        else if (!strcmp(a, "--2opt"))         o.do2opt = 1;
        else if (!strcmp(a, "--init")) {
            const char *v = NEXT();
            if      (!strcmp(v, "cw"))     o.init_random = 0;
            else if (!strcmp(v, "random")) o.init_random = 1;
            else die("--init: cw | random");
        }
        else if (!strcmp(a, "--round"))        o.rounded = 1;
        else if (!strcmp(a, "--sa-steps"))     o.sa_steps = atoi(NEXT());
        else if (!strcmp(a, "--no-sa"))        o.sa_steps = 0;
        else if (!strcmp(a, "--t0"))           o.sa_t0 = atof(NEXT());
        else if (!strcmp(a, "--tend"))         o.sa_tend = atof(NEXT());
        else if (!strcmp(a, "--sa-knn"))       o.sa_knn = atoi(NEXT());
        else if (!strcmp(a, "--t-accept"))     o.sa_chi0 = atof(NEXT());
        else if (!strcmp(a, "--t-decades"))    o.sa_decades = atof(NEXT());
        else if (!strcmp(a, "--ops")) {
            char buf[128], *c; snprintf(buf, sizeof buf, "%s", NEXT());
            for (c = buf; *c; c++) if (*c == ':' || *c == '/') *c = ',';
            o.w_rel = o.w_swap = o.w_2opt = o.w_or = 0.0;
            if (sscanf(buf, "%lf,%lf,%lf,%lf", &o.w_rel, &o.w_swap,
                       &o.w_2opt, &o.w_or) < 1)
                die("--ops: expected format r,s,t[,o] (e.g. 1,1,1,1)");
        }
        else if (!strcmp(a, "--split")) {
            const char *v = NEXT();
            if      (!strcmp(v, "off"))  o.split = 0;
            else if (!strcmp(v, "cw"))   o.split = 1;
            else if (!strcmp(v, "end"))  o.split = 2;
            else if (!strcmp(v, "both")) o.split = 3;
            else die("--split: off | cw | end | both");
        }
        else if (!strcmp(a, "--split-every")) o.split_every = atoi(NEXT());
        else if (!strcmp(a, "--split-tour")) {
            const char *v = NEXT();
            if      (!strcmp(v, "routes")) o.split_tour = 0;
            else if (!strcmp(v, "sweep"))  o.split_tour = 1;
            else if (!strcmp(v, "both"))   o.split_tour = 2;
            else die("--split-tour: routes | sweep | both");
        }
        else if (!strcmp(a, "--restarts"))     o.restarts = atoi(NEXT());
        else if (!strcmp(a, "--pick"))         o.pick_t = atoi(NEXT());
        else if (!strcmp(a, "--or-max"))       o.or_max = atoi(NEXT());
        else if (!strcmp(a, "--pick-eps"))     o.pick_eps = atof(NEXT());
        else if (!strcmp(a, "--pick-crit")) {
            const char *v = NEXT();
            if      (!strcmp(v, "lb"))  o.pick_crit = 0;
            else if (!strcmp(v, "rem")) o.pick_crit = 1;
            else if (!strcmp(v, "remnorm")) o.pick_crit = 2;
            else if (!strcmp(v, "raw")) o.pick_crit = 3;
            else die("--pick-crit: lb | rem | remnorm | raw");
        }
        else if (!strcmp(a, "--cw-alpha"))     o.cw_alpha = atof(NEXT());
        else if (!strcmp(a, "--cw-rand")) {
            const char *v = NEXT();
            if      (!strcmp(v, "off"))     o.cw_rand = 0;
            else if (!strcmp(v, "perturb")) o.cw_rand = 1;
            else if (!strcmp(v, "param"))   o.cw_rand = 2;
            else if (!strcmp(v, "both"))    o.cw_rand = 3;
            else die("--cw-rand: off | perturb | param | both");
        }
        else if (!strcmp(a, "--check"))        o.check = 1;
        else if (!strcmp(a, "--threads"))      o.threads = atoi(NEXT());
        else if (!strcmp(a, "--limit"))        o.limit = atoi(NEXT());
        else if (!strcmp(a, "--csv"))          o.csv = NEXT();
        else if (!strcmp(a, "--sol"))          o.sol = NEXT();
        else if (!strcmp(a, "--dump-bundle"))  o.dump = NEXT();
        else if (!strcmp(a, "--validate"))     o.validate = NEXT();
        else if (!strcmp(a, "--per-instance")) o.per_inst = 1;
        else if (!strcmp(a, "-q"))             o.quiet = 1;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else die("unknown option: %s (see --help)", a);
        #undef NEXT
    }
    if ((o.dir != NULL) + (o.bundle != NULL) + (o.random != 0) != 1) {
        usage();
        return 1;
    }
    if (o.cap < 0) o.cap = default_capacity(o.n);
    if (o.sa_t0 > 0 && o.sa_tend < 0) o.sa_tend = o.sa_t0 * 1e-4;
    if (o.sa_t0 > 0 && (o.sa_tend <= 0 || o.sa_tend > o.sa_t0))
        die("invalid temperatures: 0 < tend <= t0 is required");
    if (o.sa_chi0 <= 0 || o.sa_chi0 >= 1)
        die("--t-accept: 0 < chi0 < 1 is required");
    if (o.restarts < 1) die("--restarts: at least 1");
    if (o.pick_t < 0) die("--pick: >= 0 (0 = proportional, 1 = uniform)");
    if (o.pick_eps < 0) die("--pick-eps: must be positive");
    if (o.cw_alpha < 0 || o.cw_alpha >= 1) die("--cw-alpha: 0 <= alpha < 1 is required");
    if (o.or_max < 2 || o.or_max > 8) die("--or-max: between 2 and 8");
    if (o.sa_steps > 0 && o.w_rel + o.w_swap + o.w_2opt + o.w_or <= 0)
        die("--ops: at least one weight must be strictly positive");

    /* --- loading --- */
    Inst *insts = NULL;
    int M;
    double t_load = now_sec();
    if (o.dir)         insts = read_dir(o.dir, o.limit, o.rounded, &M);
    else if (o.bundle) insts = read_bundle(o.bundle, o.limit, o.rounded, &M);
    else             { M = o.m; if (o.limit > 0 && o.limit < M) M = o.limit; }
    t_load = now_sec() - t_load;

    if (o.n > (int)MAXN) die("n=%d exceeds %u", o.n, MAXN);
    if (M <= 0) die("no instance to process (-m or --limit is 0, "
                    "or the source is empty)");

    if (o.dump) {
        if (o.random) {                          /* materialise the instances */
            Inst *v = (Inst *)xmalloc((size_t)M * sizeof(Inst));
            for (int k = 0; k < M; k++) {
                inst_alloc(&v[k], o.n);
                gen_instance(&v[k], o.n, o.cap, o.seed + (uint64_t)k, o.rounded);
            }
            write_bundle(o.dump, v, M);
            for (int k = 0; k < M; k++) inst_free(&v[k]);
            free(v);
        } else write_bundle(o.dump, insts, M);
        if (!o.quiet) printf("# instances written to %s\n", o.dump);
    }
    if (o.validate) {
        int e = validate_file(o.validate, insts, M, &o);
        if (insts) { for (int k = 0; k < M; k++) inst_free(&insts[k]); free(insts); }
        return e ? 3 : 0;
    }

    int nthreads = o.threads > 0 ? o.threads : omp_get_max_threads();
    int **solbuf = NULL;
    if (o.sol) solbuf = (int **)xmalloc((size_t)M * sizeof(int *));
    omp_set_num_threads(nthreads);

    Result *R = (Result *)xmalloc((size_t)M * sizeof(Result));

    if (!o.quiet) {
        printf("# Clarke & Wright — CVRP\n");
        if (o.dir)         printf("# source     : directory %s (%d instances, loaded in %.2f s)\n", o.dir, M, t_load);
        else if (o.bundle) printf("# source     : %s (%d instances, loaded in %.2f s)\n", o.bundle, M, t_load);
        else               printf("# source     : random n=%d, m=%d, Q=%g, seed=%llu\n",
                                  o.n, M, o.cap, (unsigned long long)o.seed);
        if (o.init_random) {
            printf("# init       : random (shuffle + first fit)%s\n",
                   o.do2opt ? ", + intra-route 2-opt" : "");
        } else {
            printf("# savings    : %s", o.knn < 0 ? "exact" : (o.knn == 0 ? "auto" : "knn"));
            if (o.knn > 0) printf(" K=%d", o.knn);
            if (o.lambda != 1.0 || o.mu != 0.0) printf(", lambda=%g, mu=%g", o.lambda, o.mu);
            printf("%s\n", o.do2opt ? ", + intra-route 2-opt" : "");
        }
        if (o.sa_steps > 0) {
            double wt = o.w_rel + o.w_swap + o.w_2opt + o.w_or;
            if (o.sa_t0 > 0)
                printf("# annealing  : %d steps, T from %g to %g (geometric), K=%d\n",
                       o.sa_steps, o.sa_t0, o.sa_tend, o.sa_knn);
            else
                printf("# annealing  : %d steps, T0 calibrated (target initial "
                       "acceptance %.3g %%), %g decades, K=%d\n",
                       o.sa_steps, 100 * o.sa_chi0, o.sa_decades, o.sa_knn);
            printf("# operators  : relocate %.0f%%, swap %.0f%%, 2-opt %.0f%%, "
                   "or-opt %.0f%% (segments <= %d)\n", 100 * o.w_rel / wt,
                   100 * o.w_swap / wt, 100 * o.w_2opt / wt, 100 * o.w_or / wt,
                   o.or_max);
            printf("# selection  : %s\n", o.pick_t == 1 ? "uniform" :
                   (o.pick_t == 0 ? "proportional to regret (Fenwick)"
                                  : "tournament on regret"));
            if (o.pick_t != 1)
                { static const char *cn[4] = { "gap to the lower bound",
                        "removal gain", "normalised removal gain",
                        "raw cost of the two edges" };
                  printf("# regret     : %s\n", cn[o.pick_crit]); }
        } else printf("# annealing  : disabled\n");
        if (o.split || o.split_every) {
            static const char *sn[4] = { "off", "after C&W", "final", "C&W + final" };
            printf("# split      : %s", sn[o.split]);
            if (o.split_every) printf(", and every %d annealing steps", o.split_every);
            printf("\n");
        }
        if (o.restarts > 1) {
            static const char *cn[4] = { "none", "savings perturbation",
                                         "Yellow parameters", "both" };
            printf("# restarts   : %d per instance, randomisation: %s",
                   o.restarts, o.init_random ? "a fresh permutation"
                                             : cn[o.cw_rand]);
            if (!o.init_random && (o.cw_rand & 1)) printf(" (alpha=%g)", o.cw_alpha);
            printf("\n");
        }
        printf("# threads    : %d\n", nthreads);
        fflush(stdout);
    }

    double t0 = now_sec();

    #pragma omp parallel
    {
        WS w; memset(&w, 0, sizeof w);
        #pragma omp for schedule(dynamic, 16)
        for (int k = 0; k < M; k++) {
            const Inst *in;
            if (o.random) {
                ws_ensure(&w, o.n, 1, 1);
                gen_instance(&w.scratch, o.n, o.cap, o.seed + (uint64_t)k, o.rounded);
                in = &w.scratch;
            } else {
                in = &insts[k];
            }
            double s = now_sec();
            solve_cw(in, &w, &o, &R[k], solbuf ? &solbuf[k] : NULL,
                     o.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)k + 1);
            R[k].time_ms = (now_sec() - s) * 1e3;
        }
        ws_free(&w);
    }

    double wall = now_sec() - t0;
    if (solbuf) {
        FILE *f = fopen(o.sol, "w");
        if (!f) die("ecriture de %s : %s", o.sol, strerror(errno));
        fprintf(f, "#CWSOL 1\n#instances %d\n", M);
        fprintf(f, "#source %s\n", o.dir ? o.dir : (o.bundle ? o.bundle : "random"));
        fprintf(f, "#round %d\n", o.rounded ? 1 : 0);
        if (o.random)
            fprintf(f, "#random n=%d cap=%g seed=%llu\n",
                    o.n, o.cap, (unsigned long long)o.seed);
        fprintf(f, "#format inst <idx> <name> <n> <Q> <cost> <routes>, "
                   "then one line per route\n");
        for (int k = 0; k < M; k++) {
            fprintf(f, "inst %d %s %d %.17g %.17g %d\n", k,
                    o.random ? "random" : insts[k].name,
                    o.random ? o.n : insts[k].n,
                    o.random ? o.cap : insts[k].cap,
                    R[k].cost, R[k].routes);
            const int *b = solbuf[k];
            int first = 1;
            for (int t = 1; t <= b[0]; t++) {
                if (b[t] == 0) { fputc('\n', f); first = 1; }
                else { fprintf(f, first ? "%d" : " %d", b[t]); first = 0; }
            }
            free(solbuf[k]);
        }
        fclose(f);
        free(solbuf);
    }

    /* --- statistics --- */
    double sum = 0, sum0 = 0, sumr = 0, cpu = 0, best = 1e300, worst = -1e300, drift = 0, sacc = 0, st0 = 0, gsp = 0, sm0 = 0, sm1 = 0;
    int infeas = 0;
    double *all = (double *)xmalloc((size_t)M * sizeof(double));
    for (int k = 0; k < M; k++) {
        sum += R[k].cost; sum0 += R[k].cost0; sumr += R[k].routes; cpu += R[k].time_ms;
        if (R[k].drift > drift) drift = R[k].drift;
        sacc += R[k].acc; st0 += R[k].t0; gsp += R[k].gsplit;
        sm0 += R[k].mean0; sm1 += R[k].mean1;
        if (R[k].cost < best)  best = R[k].cost;
        if (R[k].cost > worst) worst = R[k].cost;
        if (!R[k].feasible) infeas++;
        all[k] = R[k].cost;
    }
    double mean = sum / M, var = 0;
    for (int k = 0; k < M; k++) { double d = R[k].cost - mean; var += d * d; }
    var = (M > 1) ? var / (M - 1) : 0;
    qsort(all, (size_t)M, sizeof(double), cmp_double);

    if (o.per_inst) {
        for (int k = 0; k < M; k++)
            printf("%5d %-30s init=%11.5f  annealed=%11.5f  routes=%3d  %s  %.3f ms\n",
                   k, o.random ? "(random)" : insts[k].name, R[k].cost0, R[k].cost, R[k].routes,
                   R[k].feasible ? "ok " : "NO ", R[k].time_ms);
    }
    if (o.csv) {
        FILE *f = fopen(o.csv, "w");
        if (!f) die("ecriture de %s : %s", o.csv, strerror(errno));
        fprintf(f, "instance,n,capacity,cost_init,cost_annealed,routes,max_load,feasible,time_ms\n");
        for (int k = 0; k < M; k++)
            fprintf(f, "%s,%d,%g,%.10f,%.10f,%d,%g,%d,%.4f\n",
                    o.random ? "random" : insts[k].name,
                    o.random ? o.n : insts[k].n,
                    o.random ? o.cap : insts[k].cap,
                    R[k].cost0, R[k].cost, R[k].routes, R[k].load_max,
                    R[k].feasible, R[k].time_ms);
        fclose(f);
    }

    if (!o.quiet) {
        printf("--------------------------------------------------\n");
        printf("instances        : %d  (%d infeasible)\n", M, infeas);
        printf("mean cost before annealing : %.5f\n", sum0 / M);
        printf("mean cost after annealing  : %.5f", mean);
        if (sum0 > 0) printf("   (%+.2f %%)", 100.0 * (mean - sum0 / M) / (sum0 / M));
        printf("\n");
        printf("std deviation    : %.5f\n", sqrt(var));
        if (o.restarts > 1)
            printf("mean over the %d restarts : %.5f before annealing, %.5f after "
                   "(multi-restart gain: %.5f)\n",
                   o.restarts, sm0 / M, sm1 / M, sm1 / M - mean);
        if (o.split || o.split_every)
            printf("mean Split gain  : %.5f\n", gsp / M);
        if (o.sa_steps > 0) {
            printf("annealing acceptance rate : %.1f %%", 100.0 * sacc / M);
            if (o.sa_t0 <= 0) printf("   (mean calibrated T0: %.4g)", st0 / M);
            printf("\n");
        }
        if (o.check || drift > 1e-6)
            printf("max incremental-cost drift : %.3e\n", drift);
        printf("median / min / max : %.5f / %.5f / %.5f\n", all[M / 2], best, worst);
        printf("routes (mean)    : %.3f\n", sumr / M);
        printf("total time       : %.3f s  (wall), %.3f s (cumulative CPU)\n", wall, cpu / 1e3);
        printf("time / instance  : %.4f ms\n", cpu / M);
        printf("throughput       : %.1f instances/s\n", M / wall);
    } else {
        printf("%.6f\n", mean);
    }

    free(all); free(R);
    if (insts) { for (int k = 0; k < M; k++) inst_free(&insts[k]); free(insts); }
    return infeas ? 2 : 0;
}
