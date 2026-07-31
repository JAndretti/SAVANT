/* ============================================================================
 *  trace.c — instrumented single-instance driver for SAVANT.
 *
 *  Records what `cw` computes but does not report: the cost trajectory inside
 *  one annealing run, and per-operator draw / acceptance counts.
 *
 *  cw.c is NOT modified. This file #includes it, which works because every
 *  function there is `static` — including the whole translation unit makes them
 *  all visible here. cw.c's own `main` is renamed out of the way by the #define
 *  below, so this file can supply its own.
 *
 *  Consequences of that choice:
 *    - zero duplication: the savings construction, kNN, workspace, operators,
 *      RNG and Split are the ones the real solver uses, not copies;
 *    - the traced loop consumes the RNG in exactly the same order as `anneal`
 *      (calibration draws included), so with the same seed and options the
 *      final cost matches `cw` bit for bit — which is the check that the trace
 *      describes the real search and not a lookalike.
 *
 *  Scope, by design: one instance, one run, no multi-restart.
 *
 *  Build :  make trace          ->  ./cw_trace
 *  Help  :  ./cw_trace --help
 * ==========================================================================*/

#define main cw_main_unused
#include "cw.c"
#undef main

#include <sys/stat.h>

/* ------------------------------------------------------------------ output */

typedef struct {
    long   draws[4];        /* moves drawn per operator                      */
    long   acc[4];          /* of those, accepted                            */
    double gain[4];         /* summed delta of the accepted ones (<0 = good) */
    double best_at[4];      /* cost improvement that set a new best          */
    long   newbest[4];      /* number of times this operator set a new best  */
} OpStats;

static const char *OPNAME[4] = { "relocate", "swap", "2-opt", "or-opt" };

/* mkdir -p, so `results/` need not already exist */
static void ensure_dir(const char *path)
{
    char buf[4096];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; p++)
        if (*p == '/') { *p = 0; mkdir(buf, 0777); *p = '/'; }
    if (mkdir(buf, 0777) != 0 && errno != EEXIST)
        die("creating %s: %s", path, strerror(errno));
}

/* ------------------------------------------------------- random construction
 * A *feasible* random start, not a uniformly random one: shuffle the customers
 * and cut the sequence whenever the next one would overflow the vehicle. That
 * keeps the comparison against Clarke & Wright about solution quality rather
 * than about repairing infeasibility — which the annealer structurally cannot
 * do anyway, since every operator rejects a capacity violation.
 *
 * First-fit on a random permutation also lands on a route count close to C&W's,
 * so the two starts are comparable in structure and differ mainly in quality. */

static double random_init(const Inst *in, WS *w, Sol *S, uint64_t seed)
{
    const int n = in->n;
    int *perm = w->bufA;                       /* sized n+2 by ws_ensure */
    for (int i = 0; i < n; i++) perm[i] = i + 1;

    Rng r; rng_seed(&r, seed);
    for (int i = n - 1; i > 0; i--) {          /* Fisher-Yates */
        int j = rng_idx(&r, i + 1);
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }

    int nr = 0, start = 0;
    double load = 0.0;
    for (int i = 0; i < n; i++) {
        if (i > start && load + in->dem[perm[i]] > in->cap + EPS) {
            route_set(S, in, nr++, perm + start, i - start);
            start = i; load = 0.0;
        }
        load += in->dem[perm[i]];
    }
    if (start < n) route_set(S, in, nr++, perm + start, n - start);
    S->R = nr;
    return sol_cost(in, S);
}

/* ------------------------------------------------- instrumented annealing
 * A copy of anneal() with three additions: the operator is drawn here rather
 * than inside sa_draw so we know which one ran; acceptance is read from the
 * solver's own counter (w->acc increments only on acceptance, so this is exact
 * rather than inferred from delta == 0); and a row is written every `every`
 * steps. Everything else — the threshold setup, the calibration, the geometric
 * schedule, the best-so-far snapshot — is line for line what cw.c does. */

static double trace_anneal(const Inst *in, WS *w, const Opts *o, Sol *S,
                           double cur, uint64_t seed,
                           FILE *tf, long every, OpStats *st, double *t0_out)
{
    const int n = S->n, R = S->R, steps = o->sa_steps;
    if (steps <= 0) return cur;

    int K = o->sa_knn; if (K > n - 1) K = n - 1; if (K < 0) K = 0;
    w->pick_t = (K > 0) ? o->pick_t : 1;
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
        w->eps0 = o->pick_eps * m / (2.0 * n);
    } else if (w->crit == 0 || w->crit == 2) w->pick_t = 1;
    else w->eps0 = o->pick_eps * cur / (double)(n + R);
    inc_build(in, w, S);

    Rng rng; rng_seed(&rng, seed);
    const size_t zi = (size_t)(2 * n + 4) * sizeof(int);
    const size_t zd = (size_t)(n + 2) * sizeof(double);
    int bR = R;

    double T0 = o->sa_t0, Tend = o->sa_tend;
    if (T0 <= 0.0) {
        T0 = calibrate_T0(in, w, o, S, K, &rng);
        if (T0 <= 0.0) return cur;                 /* frozen solution */
        Tend = T0 * pow(10.0, -o->sa_decades);
    }
    *t0_out = T0;
    double T = T0;
    const double alpha = (steps > 1) ? pow(Tend / T0, 1.0 / (steps - 1)) : 1.0;
    double best = cur;
    memcpy(w->b_nxt, S->nxt, zi); memcpy(w->b_prv, S->prv, zi);
    memcpy(w->b_rid, S->rid, zi); memcpy(w->b_load, S->load, zd);

    if (tf) fprintf(tf, "step,T,cur,best,op,accepted,delta,routes\n");

    for (int it = 0; it < steps; it++) {
        /* same threshold comparison sa_draw() makes, so the RNG stream and the
           resulting trajectory are identical to a plain cw run */
        uint32_t z = rng32(&rng);
        int op = (z < w->th1) ? 0 : (z < w->th2) ? 1 : (z < w->th3) ? 2 : 3;

        long acc_before = w->acc;
        double d;
        switch (op) {
            case 0:  d = mv_relocate(in, w, S, K, T, &rng, 0); break;
            case 1:  d = mv_swap    (in, w, S, K, T, &rng, 0); break;
            case 2:  d = mv_2opt    (in, w, S, K, T, &rng, 0); break;
            default: d = mv_oropt   (in, w, S, K, T, &rng, 0); break;
        }
        int accepted = (w->acc > acc_before);

        st->draws[op]++;
        if (accepted) { st->acc[op]++; st->gain[op] += d; }
        cur += d;

        if (cur < best - 1e-12) {
            if (accepted) { st->newbest[op]++; st->best_at[op] += best - cur; }
            best = cur;
            bR = S->R;
            memcpy(w->b_nxt, S->nxt, zi); memcpy(w->b_prv, S->prv, zi);
            memcpy(w->b_rid, S->rid, zi); memcpy(w->b_load, S->load, zd);
        }
        if (tf && (it % every == 0 || it == steps - 1)) {
            int live = 0;
            for (int r = 0; r < S->R; r++) if (S->nxt[n + 1 + r] != n + 1 + r) live++;
            fprintf(tf, "%d,%.10g,%.10f,%.10f,%d,%d,%.10g,%d\n",
                    it, T, cur, best, op, accepted, d, live);
        }
        T *= alpha;
    }

    memcpy(S->nxt, w->b_nxt, zi); memcpy(S->prv, w->b_prv, zi);
    memcpy(S->rid, w->b_rid, zi); memcpy(S->load, w->b_load, zd);
    S->R = bR;
    inc_build(in, w, S);
    return best;
}

/* --------------------------------------------------------------------- CLI */

static void trace_usage(void)
{
    printf(
"trace.c — instrumented single-instance run (cw.c is not modified)\n"
"\n"
"Usage: cw_trace [source] [options]\n"
"\n"
"Source (exactly one):\n"
"  --bundle FILE      .cvrpb bundle; --index K picks the instance (default 0)\n"
"  --random           generate one instance (-n N --cap C --seed S)\n"
"\n"
"Initial solution:\n"
"  --init cw|random   cw (default) = Clarke & Wright, as the solver does;\n"
"                     random = shuffle the customers and cut into routes at\n"
"                     the capacity limit. Isolates what the construction buys.\n"
"\n"
"Annealing (same meaning and defaults as cw):\n"
"  --sa-steps N       default 100000\n"
"  --sa-knn K         default 20\n"
"  --ops r,s,t,o      default 1,1,1,0\n"
"  --or-max L         default 3\n"
"  --pick T           default 2        --pick-crit lb|rem|remnorm|raw\n"
"  --pick-eps E       default 0.3\n"
"  --t-accept X       default 0.001    --t-decades D   default 2\n"
"  --t0 T --tend T    fix the schedule by hand\n"
"  --knn K | --exact  savings list size for the construction\n"
"  --seed S           default 42\n"
"\n"
"Output (one directory per trace, like run.py):\n"
"  --out DIR          default results/trace_<timestamp>[_<name>]\n"
"  --name TAG         readable suffix on the default directory\n"
"  --every N          record one row every N steps      (default 1)\n"
"  --no-csv           summary only, no per-step file\n"
"\n"
"The directory receives trace.csv and trace.json; analyze.py adds\n"
"analysis.png:  python3 tools/analyze.py --trace <DIR>\n"
"\n"
"One instance, one run, no multi-restart by construction.\n");
}

int main(int argc, char **argv)
{
    Opts o;
    memset(&o, 0, sizeof o);
    o.n = 100; o.m = 1; o.seed = 42; o.cap = -1;
    o.lambda = 1.0; o.mu = 0.0; o.threads = 1; o.knn = 0;
    o.sa_steps = 100000; o.sa_t0 = -1.0; o.sa_tend = -1.0;
    o.sa_chi0 = 0.001; o.sa_decades = 2.0;
    o.restarts = 1; o.cw_rand = 0; o.cw_alpha = 0.03;
    o.pick_t = 2; o.pick_eps = 0.3; o.pick_crit = 0; o.sa_knn = 20;
    o.w_rel = 1.0; o.w_swap = 1.0; o.w_2opt = 1.0; o.w_or = 0.0; o.or_max = 3;

    const char *out = NULL, *name = NULL;
    int index = 0, want_csv = 1, init_random = 0;
    long every = 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT() (i + 1 < argc ? argv[++i] : (die("option %s: missing argument", a), ""))
        if      (!strcmp(a, "--bundle"))    o.bundle = NEXT();
        else if (!strcmp(a, "--random"))    o.random = 1;
        else if (!strcmp(a, "--index"))     index = atoi(NEXT());
        else if (!strcmp(a, "-n"))          o.n = atoi(NEXT());
        else if (!strcmp(a, "--cap"))       o.cap = atof(NEXT());
        else if (!strcmp(a, "--seed"))      o.seed = strtoull(NEXT(), NULL, 10);
        else if (!strcmp(a, "--knn"))       o.knn = atoi(NEXT());
        else if (!strcmp(a, "--exact"))     o.knn = -1;
        else if (!strcmp(a, "--lambda"))    o.lambda = atof(NEXT());
        else if (!strcmp(a, "--mu"))        o.mu = atof(NEXT());
        else if (!strcmp(a, "--round"))     o.rounded = 1;
        else if (!strcmp(a, "--sa-steps"))  o.sa_steps = atoi(NEXT());
        else if (!strcmp(a, "--sa-knn"))    o.sa_knn = atoi(NEXT());
        else if (!strcmp(a, "--or-max"))    o.or_max = atoi(NEXT());
        else if (!strcmp(a, "--pick"))      o.pick_t = atoi(NEXT());
        else if (!strcmp(a, "--pick-eps"))  o.pick_eps = atof(NEXT());
        else if (!strcmp(a, "--t-accept"))  o.sa_chi0 = atof(NEXT());
        else if (!strcmp(a, "--t-decades")) o.sa_decades = atof(NEXT());
        else if (!strcmp(a, "--t0"))        o.sa_t0 = atof(NEXT());
        else if (!strcmp(a, "--tend"))      o.sa_tend = atof(NEXT());
        else if (!strcmp(a, "--out"))       out = NEXT();
        else if (!strcmp(a, "--name"))      name = NEXT();
        else if (!strcmp(a, "--init")) {
            const char *v = NEXT();
            if      (!strcmp(v, "cw"))     init_random = 0;
            else if (!strcmp(v, "random")) init_random = 1;
            else die("--init: cw | random");
        }
        else if (!strcmp(a, "--every"))     every = atol(NEXT());
        else if (!strcmp(a, "--no-csv"))    want_csv = 0;
        else if (!strcmp(a, "--ops")) {
            char buf[128], *c; snprintf(buf, sizeof buf, "%s", NEXT());
            for (c = buf; *c; c++) if (*c == ':' || *c == '/') *c = ',';
            o.w_rel = o.w_swap = o.w_2opt = o.w_or = 0.0;
            if (sscanf(buf, "%lf,%lf,%lf,%lf", &o.w_rel, &o.w_swap,
                       &o.w_2opt, &o.w_or) < 1)
                die("--ops: expected format r,s,t[,o]");
        }
        else if (!strcmp(a, "--pick-crit")) {
            const char *v = NEXT();
            if      (!strcmp(v, "lb"))      o.pick_crit = 0;
            else if (!strcmp(v, "rem"))     o.pick_crit = 1;
            else if (!strcmp(v, "remnorm")) o.pick_crit = 2;
            else if (!strcmp(v, "raw"))     o.pick_crit = 3;
            else die("--pick-crit: lb | rem | remnorm | raw");
        }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { trace_usage(); return 0; }
        else die("unknown option: %s (see --help)", a);
        #undef NEXT
    }
    if ((o.bundle != NULL) + (o.random != 0) != 1) { trace_usage(); return 1; }
    if (every < 1) die("--every: at least 1");
    if (o.cap < 0) o.cap = default_capacity(o.n);
    if (o.sa_t0 > 0 && o.sa_tend < 0) o.sa_tend = o.sa_t0 * 1e-4;
    if (o.or_max < 2 || o.or_max > 8) die("--or-max: between 2 and 8");

    /* ---------------------------------------------------------- instance */
    Inst inst, *loaded = NULL;
    int count = 0;
    if (o.bundle) {
        loaded = read_bundle(o.bundle, index + 1, o.rounded, &count);
        if (index >= count) die("--index %d: bundle holds %d instance(s)", index, count);
        inst = loaded[index];
    } else {
        inst_alloc(&inst, o.n);
        gen_instance(&inst, o.n, o.cap, o.seed + (uint64_t)index, o.rounded);
    }
    const int n = inst.n;

    /* Seeds derived exactly as cw does, so the run is the one cw would make
       for instance `index` of this source with this --seed. */
    const uint64_t inst_seed = o.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)index + 1;
    const uint64_t anneal_seed = inst_seed ^ 0x5DEECE66DULL;

    /* ------------------------------------------------- initial solution */
    WS w; memset(&w, 0, sizeof w);
    Result res; memset(&res, 0, sizeof res);
    Sol S;
    double cost0, t_cw;

    if (init_random) {
        double dmax = 0.0;
        for (int i = 1; i <= n; i++) if (inst.dem[i] > dmax) dmax = inst.dem[i];
        if (dmax > inst.cap + EPS)
            die("instance is infeasible (demand %g > capacity %g): "
                "no feasible random start exists", dmax, inst.cap);
        int K = o.sa_knn > 0 ? o.sa_knn : 1;
        ws_ensure(&w, n, K, 1);
        w.knn_k = 0;                    /* force the kNN lists to be built */
        S.n = n; S.nxt = w.s_nxt; S.prv = w.s_prv;
        S.rid = w.s_rid; S.load = w.s_load;
        double t0c = now_sec();
        cost0 = random_init(&inst, &w, &S, inst_seed ^ 0xA5A5A5A5A5A5A5A5ULL);
        t_cw = now_sec() - t0c;
        res.routes = S.R;
    } else {
        Opts build = o;
        build.sa_steps = 0;             /* construction only */
        build.split = 0; build.split_every = 0;
        int *flat = NULL;
        double t0c = now_sec();
        solve_cw(&inst, &w, &build, &res, &flat, inst_seed);
        t_cw = now_sec() - t0c;

        /* rebuild a linked Sol from the flat [len, c.., 0, c.., 0] emitted */
        S.n = n; S.nxt = w.s_nxt; S.prv = w.s_prv;
        S.rid = w.s_rid; S.load = w.s_load;
        int r = 0;
        for (int t = 1; t <= flat[0]; ) {
            int len = 0;
            while (t + len <= flat[0] && flat[t + len] != 0) len++;
            if (len) route_set(&S, &inst, r++, flat + t, len);
            t += len + 1;
        }
        S.R = r;
        free(flat);
        cost0 = sol_cost(&inst, &S);
    }
    const int routes0 = S.R;

    /* --------------------------------------------------------- annealing */
    char outdir[4096], path[4200];
    if (out) {
        snprintf(outdir, sizeof outdir, "%s", out);
    } else {
        time_t now = time(NULL);
        struct tm tmv; localtime_r(&now, &tmv);
        char stamp[32];
        strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tmv);
        if (name) snprintf(outdir, sizeof outdir, "results/trace_%s_%s", stamp, name);
        else      snprintf(outdir, sizeof outdir, "results/trace_%s", stamp);
    }
    ensure_dir(outdir);

    FILE *tf = NULL;
    if (want_csv) {
        snprintf(path, sizeof path, "%s/trace.csv", outdir);
        tf = fopen(path, "w");
        if (!tf) die("writing %s: %s", path, strerror(errno));
    }
    OpStats st; memset(&st, 0, sizeof st);
    double t0_used = 0.0;
    double t_sa0 = now_sec();
    double tracked = trace_anneal(&inst, &w, &o, &S, cost0, anneal_seed,
                                  tf, every, &st, &t0_used);
    double t_sa = now_sec() - t_sa0;
    if (tf) fclose(tf);

    double final = sol_cost(&inst, &S);
    long total_draws = 0, total_acc = 0;
    for (int i = 0; i < 4; i++) { total_draws += st.draws[i]; total_acc += st.acc[i]; }

    /* ------------------------------------------------------------ summary */
    snprintf(path, sizeof path, "%s/trace.json", outdir);
    FILE *jf = fopen(path, "w");
    if (!jf) die("writing %s: %s", path, strerror(errno));
    fprintf(jf, "{\n");
    fprintf(jf, "  \"source\": \"%s\",\n", o.bundle ? o.bundle : "random");
    fprintf(jf, "  \"instance\": {\"index\": %d, \"name\": \"%s\", \"n\": %d, \"capacity\": %g},\n",
            index, inst.name, n, inst.cap);
    fprintf(jf, "  \"init\": \"%s\",\n", init_random ? "random" : "cw");
    fprintf(jf, "  \"seed\": %llu,\n", (unsigned long long)o.seed);
    fprintf(jf, "  \"steps\": %d,\n", o.sa_steps);
    fprintf(jf, "  \"every\": %ld,\n", every);
    fprintf(jf, "  \"sa_knn\": %d,\n", o.sa_knn);
    fprintf(jf, "  \"pick\": %d,\n", o.pick_t);
    fprintf(jf, "  \"ops\": [%g, %g, %g, %g],\n",
            o.w_rel, o.w_swap, o.w_2opt, o.w_or);
    fprintf(jf, "  \"t0\": %.10g,\n", t0_used);
    fprintf(jf, "  \"tend\": %.10g,\n",
            o.sa_t0 > 0 ? o.sa_tend : t0_used * pow(10.0, -o.sa_decades));
    fprintf(jf, "  \"cost_init\": %.10f,\n", cost0);
    fprintf(jf, "  \"routes_init\": %d,\n", routes0);
    fprintf(jf, "  \"cost_final\": %.10f,\n", final);
    fprintf(jf, "  \"cost_tracked\": %.10f,\n", tracked);
    fprintf(jf, "  \"drift\": %.3e,\n", fabs(final - tracked));
    fprintf(jf, "  \"routes\": %d,\n", res.routes);
    fprintf(jf, "  \"time_cw_ms\": %.4f,\n", t_cw * 1e3);
    fprintf(jf, "  \"time_sa_ms\": %.4f,\n", t_sa * 1e3);
    fprintf(jf, "  \"draws\": %ld,\n", total_draws);
    fprintf(jf, "  \"accepted\": %ld,\n", total_acc);
    fprintf(jf, "  \"operators\": [\n");
    for (int i = 0; i < 4; i++)
        fprintf(jf, "    {\"name\": \"%s\", \"weight\": %g, \"draws\": %ld, "
                    "\"accepted\": %ld, \"accept_rate\": %.6f, "
                    "\"sum_delta\": %.10f, \"new_best\": %ld}%s\n",
                OPNAME[i],
                i == 0 ? o.w_rel : i == 1 ? o.w_swap : i == 2 ? o.w_2opt : o.w_or,
                st.draws[i], st.acc[i],
                st.draws[i] ? (double)st.acc[i] / (double)st.draws[i] : 0.0,
                st.gain[i], st.newbest[i], i < 3 ? "," : "");
    fprintf(jf, "  ]\n}\n");
    fclose(jf);

    /* ------------------------------------------------------------- stdout */
    printf("instance %d (%s): n=%d Q=%g\n", index, inst.name, n, inst.cap);
    printf("%-14s: %.6f  (%d routes, %.3f ms)\n",
           init_random ? "random start" : "C&W cost",
           cost0, routes0, t_cw * 1e3);
    printf("after %d steps: %.6f   (%+.2f %%, %.1f ms)\n", o.sa_steps, final,
           100.0 * (final - cost0) / cost0, t_sa * 1e3);
    printf("T0 %.5g -> Tend %.5g   drift %.2e\n", t0_used,
           o.sa_t0 > 0 ? o.sa_tend : t0_used * pow(10.0, -o.sa_decades),
           fabs(final - tracked));
    printf("\n%-10s %10s %10s %8s %12s %9s\n",
           "operator", "draws", "accepted", "rate", "sum delta", "new best");
    for (int i = 0; i < 4; i++) {
        if (!st.draws[i]) continue;
        printf("%-10s %10ld %10ld %7.2f%% %12.5f %9ld\n",
               OPNAME[i], st.draws[i], st.acc[i],
               100.0 * (double)st.acc[i] / (double)st.draws[i],
               st.gain[i], st.newbest[i]);
    }
    printf("%-10s %10ld %10ld %7.2f%%\n", "total", total_draws, total_acc,
           total_draws ? 100.0 * (double)total_acc / (double)total_draws : 0.0);
    printf("\n-> %s/  (trace.json%s)\n", outdir,
           want_csv ? " + trace.csv" : "");

    ws_free(&w);
    if (loaded) { for (int k = 0; k < count; k++) inst_free(&loaded[k]); free(loaded); }
    else inst_free(&inst);
    return 0;
}
