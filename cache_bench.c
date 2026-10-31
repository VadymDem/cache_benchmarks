#define _CRT_SECURE_NO_WARNINGS
#if defined(_WIN32) && !defined(_WIN32_WINNT)
  #define _WIN32_WINNT 0x0A00
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <math.h>
#include <time.h>

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <process.h>
  #include <direct.h>
  typedef unsigned (__stdcall *thread_fn)(void*);
  typedef HANDLE th_t;
  #define THREAD_FCN unsigned __stdcall
#else
  #include <pthread.h>
  #include <unistd.h>
  #include <sys/stat.h>
  #include <sched.h>
  typedef void* (*thread_fn)(void*);
  typedef pthread_t th_t;
  #define THREAD_FCN void*
#endif

#define CACHELINE 64
#define MAX_THREADS 64
#define TILE 64

/* ============================ Вивід =============================== */
static FILE* g_out = NULL;

static void out_printf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    if (g_out) {
        va_list ap2;
        va_start(ap2, fmt);
        vfprintf(g_out, fmt, ap2);
        va_end(ap2);
    }
    fflush(stdout);
    if (g_out) fflush(g_out);
}

/* =========================== Час ================================== */
static double now_sec(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int inited = 0;
    if (!inited) { QueryPerformanceFrequency(&freq); inited = 1; }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* ========================== Потоки ================================ */
static th_t thread_start(thread_fn fn, void* arg) {
#if defined(_WIN32)
    uintptr_t h = _beginthreadex(NULL, 0, fn, arg, 0, NULL);
    return (th_t)h;
#else
    pthread_t t;
    pthread_create(&t, NULL, fn, arg);
    return t;
#endif
}

static void thread_join(th_t t) {
#if defined(_WIN32)
    WaitForSingleObject((HANDLE)t, INFINITE);
    CloseHandle((HANDLE)t);
#else
    pthread_join(t, NULL);
#endif
}

static void cpu_yield(void) {
#if defined(_WIN32)
    SwitchToThread();
#else
    sched_yield();
#endif
}

/* ========================== М'ютекс =============================== */
typedef struct {
#if defined(_WIN32)
    CRITICAL_SECTION cs;
#else
    pthread_mutex_t m;
#endif
} mutex_t;

static void mutex_init(mutex_t* m) {
#if defined(_WIN32)
    InitializeCriticalSection(&m->cs);
#else
    pthread_mutex_init(&m->m, NULL);
#endif
}
static void mutex_lock(mutex_t* m) {
#if defined(_WIN32)
    EnterCriticalSection(&m->cs);
#else
    pthread_mutex_lock(&m->m);
#endif
}
static void mutex_unlock(mutex_t* m) {
#if defined(_WIN32)
    LeaveCriticalSection(&m->cs);
#else
    pthread_mutex_unlock(&m->m);
#endif
}
static void mutex_destroy(mutex_t* m) {
#if defined(_WIN32)
    DeleteCriticalSection(&m->cs);
#else
    pthread_mutex_destroy(&m->m);
#endif
}

/* ===================== Бар'єр старту ============================== */
typedef struct {
    atomic_int count;
    atomic_int go;
    int total;
} barrier_t;

static void barrier_init(barrier_t* b, int total) {
    atomic_store(&b->count, 0);
    atomic_store(&b->go, 0);
    b->total = total;
}

static void barrier_wait(barrier_t* b) {
    atomic_fetch_add_explicit(&b->count, 1, memory_order_acq_rel);
    while (!atomic_load_explicit(&b->go, memory_order_acquire))
        cpu_yield();
}

static void barrier_release(barrier_t* b) {
    while (atomic_load_explicit(&b->count, memory_order_acquire) < b->total)
        cpu_yield();
    atomic_store_explicit(&b->go, 1, memory_order_release);
}

/* ====================== Виділення пам'яті ========================= */
static void* xalloc(size_t bytes, size_t align) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, align);
#else
    void* p = NULL;
    if (posix_memalign(&p, align, bytes) != 0) return NULL;
    return p;
#endif
}

static void xfree(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ======================= Генератор ================================ */
static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;

static uint64_t rng_next(void) {
    uint64_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    g_rng = x;
    return x;
}

static double rng_double(void) {
    return ((double)(rng_next() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
}

/* ==================== Налаштування з env ========================== */
typedef struct {
    int threads;
    int matmul_n;
    int rows_n;
    int matmul_reps;
    uint64_t fs_iters;
    uint64_t race_iters;
    int max_mb;
    int stride_mb;
    uint64_t lat_accesses;
} cfg_t;

static int env_int(const char* name, int defv) {
    const char* s = getenv(name);
    if (!s || !*s) return defv;
    long v = strtol(s, NULL, 10);
    return (int)v;
}

static uint64_t env_u64(const char* name, uint64_t defv) {
    const char* s = getenv(name);
    if (!s || !*s) return defv;
    return (uint64_t)strtoull(s, NULL, 10);
}

static cfg_t load_cfg(void) {
    cfg_t c;
    c.threads = env_int("LAB3_THREADS", 8);
    if (c.threads < 1) c.threads = 1;
    if (c.threads > MAX_THREADS) c.threads = MAX_THREADS;
    c.matmul_n = env_int("LAB3_MATMUL_N", 1024);
    if (c.matmul_n < 16) c.matmul_n = 16;
    c.rows_n = env_int("LAB3_ROWS_N", 2048);
    if (c.rows_n < 64) c.rows_n = 64;
    c.matmul_reps = env_int("LAB3_MATMUL_REPS", 2);
    if (c.matmul_reps < 1) c.matmul_reps = 1;
    c.fs_iters = env_u64("LAB3_FS_ITERS", 4000000ULL);
    c.race_iters = env_u64("LAB3_RACE_ITERS", 1000000ULL);
    c.max_mb = env_int("LAB3_MAX_MB", 128);
    c.stride_mb = env_int("LAB3_STRIDE_MB", 64);
    c.lat_accesses = env_u64("LAB3_LAT_ACCESSES", 0);
    if (c.max_mb < 1) c.max_mb = 1;
    if (c.stride_mb < 1) c.stride_mb = 1;
    return c;
}

/* ================================================================== */
/* 1. Ієрархія пам'яті: послідовний і випадковий доступ              */
/* ================================================================== */

/* Односпрямований цикл через усі елементи — кожен наступний індекс
   залежить від попереднього, тому апаратний префетчер безсилий. */
static uint64_t* build_cycle(size_t n) {
    uint64_t* next = (uint64_t*)xalloc(n * sizeof(uint64_t), 64);
    uint64_t* perm = (uint64_t*)xalloc(n * sizeof(uint64_t), 64);
    if (!next || !perm) { xfree(next); xfree(perm); return NULL; }

    for (size_t i = 0; i < n; i++) perm[i] = i;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rng_next() % (i + 1));
        uint64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (size_t k = 0; k < n; k++)
        next[perm[k]] = perm[(k + 1) % n];

    xfree(perm);
    return next;
}

typedef struct {
    double sec;
    uint64_t bytes;
    uint64_t accesses;
    double checksum;
} mem_result_t;

static mem_result_t measure_seq(uint64_t* a, size_t n, int reps) {
    volatile uint64_t sink = 0;
    uint64_t sum = 0;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        uint64_t s = 0;
        for (size_t i = 0; i < n; i++) s += a[i];
        sink += s;
        sum += s;
    }
    double t1 = now_sec();
    mem_result_t res;
    res.sec = t1 - t0;
    res.bytes = (uint64_t)n * sizeof(uint64_t) * (uint64_t)reps;
    res.accesses = (uint64_t)n * (uint64_t)reps;
    res.checksum = (double)(sum + (uint64_t)sink);
    return res;
}

static mem_result_t measure_random(const uint64_t* next, size_t n, uint64_t accesses) {
    volatile uint64_t sink = 0;
    size_t idx = 0;
    uint64_t sum = 0;
    /* прогрів */
    for (int i = 0; i < 100000; i++) idx = (size_t)next[idx];
    double t0 = now_sec();
    for (uint64_t i = 0; i < accesses; i++) {
        idx = (size_t)next[idx];
        sum += (uint64_t)idx;
    }
    double t1 = now_sec();
    sink += sum;
    mem_result_t res;
    res.sec = t1 - t0;
    res.bytes = accesses * sizeof(uint64_t);
    res.accesses = accesses;
    res.checksum = (double)(sum + (uint64_t)sink);
    (void)n;
    return res;
}

static void section_latency(const cfg_t* cfg) {
    static const int sizes_kb[] = {16, 32, 64, 128, 256, 512, 1024, 2048,
                                   4096, 8192, 16384, 32768, 65536, 131072};
    int nsizes = (int)(sizeof(sizes_kb) / sizeof(sizes_kb[0]));

    out_printf("\n# == 1) Memory hierarchy: sequential vs random ==\n");
    out_printf("# size_KB, bytes, seq_GBps, seq_ns_op, rnd_ns_op, rnd_Mops\n");

    for (int s = 0; s < nsizes; s++) {
        int kb = sizes_kb[s];
        if (kb > cfg->max_mb * 1024) break;
        size_t n = ((size_t)kb * 1024) / sizeof(uint64_t);
        if (n < 2) continue;

        uint64_t* a = (uint64_t*)xalloc(n * sizeof(uint64_t), 64);
        if (!a) { out_printf("# alloc failed for %d KB\n", kb); continue; }
        for (size_t i = 0; i < n; i++) a[i] = rng_next();

        size_t bytes = n * sizeof(uint64_t);
        int reps = (int)((512ULL * 1024 * 1024) / bytes);
        if (reps < 3) reps = 3;
        if (reps > 65536) reps = 65536;

        mem_result_t seq = measure_seq(a, n, reps);

        /* для випадкового доступу перетворюємо масив у цикл;
           самі дані не потрібні, важлива лише адреса */
        uint64_t* cycle = build_cycle(n);
        if (!cycle) { xfree(a); continue; }
        uint64_t acc = cfg->lat_accesses;
        if (acc == 0) acc = (n < 2000000) ? 4000000ULL : 2000000ULL;
        if (acc > n * 64ULL) acc = n * 64ULL;
        if (acc < 1000) acc = 1000;

        mem_result_t rnd = measure_random(cycle, n, acc);

        double seq_gbps = (double)seq.bytes / seq.sec / 1e9;
        double seq_ns = seq.sec / (double)seq.accesses * 1e9;
        double rnd_ns = rnd.sec / (double)rnd.accesses * 1e9;

        out_printf("LATSEQ,%d,%zu,%.3f,%.4f,%.4f,%.3f\n",
                   kb, bytes, seq_gbps, seq_ns, rnd_ns,
                   rnd_ns > 0 ? (1000.0 / rnd_ns) : 0.0);

        xfree(cycle);
        xfree(a);
    }
}

/* ================================================================== */
/* 2. Крок доступу та розмір кеш-лінії                              */
/* ================================================================== */
static void section_stride(const cfg_t* cfg) {
    static const int strides[] = {8, 16, 32, 48, 64, 80, 96, 128, 192, 256,
                                  384, 512, 1024, 2048, 4096};
    int nstrides = (int)(sizeof(strides) / sizeof(strides[0]));
    double ns_ops[16];

    size_t n = ((size_t)cfg->stride_mb * 1024 * 1024) / sizeof(uint64_t);
    if (n < 1024) n = 1024;
    uint64_t* a = (uint64_t*)xalloc(n * sizeof(uint64_t), 64);
    if (!a) { out_printf("# stride: alloc failed\n"); return; }
    for (size_t i = 0; i < n; i++) a[i] = i;

    out_printf("\n# == 2) Stride sweep (buffer %d MB) ==\n", cfg->stride_mb);
    out_printf("# stride_B, accesses, ns_op, payload_GBps, line_GBps\n");

    double min_ns = 1e300;
    int estimate = 0;
    for (int s = 0; s < nstrides; s++) {
        int stride = strides[s];
        size_t step = (size_t)stride / sizeof(uint64_t);
        if (step == 0) continue;
        size_t count = n / step;
        if (count < 16) continue;

        int reps = (int)((64ULL * 1024 * 1024) / ((uint64_t)count * sizeof(uint64_t)));
        if (reps < 1) reps = 1;
        if (reps > 10000) reps = 10000;

        volatile uint64_t sink = 0;
        double t0 = now_sec();
        uint64_t sum = 0;
        for (int r = 0; r < reps; r++) {
            uint64_t acc = 0;
            for (size_t i = 0; i < count; i++) acc += a[i * step];
            sum += acc;
        }
        double t1 = now_sec();
        sink += sum;

        double sec = t1 - t0;
        double ns_op = sec / ((double)count * reps) * 1e9;
        double payload = (double)count * reps * sizeof(uint64_t) / sec / 1e9;
        double lines = (double)count * reps * CACHELINE / sec / 1e9;
        out_printf("STRIDE,%d,%zu,%.3f,%.3f,%.3f\n",
                   stride, count * reps, ns_op, payload, lines);
        (void)sink;
        ns_ops[s] = ns_op;
        if (ns_op < min_ns) min_ns = ns_op;
    }

    /* Оцінка лінії: перший степінь двійки (>= 32 B), після якого ціна
       доступу зростає щонайменше утричі порівняно з мінімумом. */
    for (int s = 0; s < nstrides && strides[s] <= 512; s++) {
        int stride = strides[s];
        if (stride < 32 || (stride & (stride - 1)) != 0) continue;
        if (ns_ops[s] > min_ns * 3.0) { estimate = stride; break; }
    }
    out_printf("CACHELINE,estimate,%d,min_ns,%.3f\n", estimate, min_ns);
    xfree(a);
}

/* ================================================================== */
/* 3. Множення матриць                                               */
/* ================================================================== */
typedef void (*mm_fn)(const double*, const double*, double*, int);

static void mm_ijk(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
    }
}

static void mm_ikj(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double aik = A[(size_t)i * n + k];
            const double* brow = B + (size_t)k * n;
            double* crow = C + (size_t)i * n;
            for (int j = 0; j < n; j++)
                crow[j] += aik * brow[j];
        }
    }
}

static void mm_tiled(const double* A, const double* B, double* C, int n) {
    for (int ii = 0; ii < n; ii += TILE) {
        int iend = ii + TILE < n ? ii + TILE : n;
        for (int kk = 0; kk < n; kk += TILE) {
            int kend = kk + TILE < n ? kk + TILE : n;
            for (int jj = 0; jj < n; jj += TILE) {
                int jend = jj + TILE < n ? jj + TILE : n;
                for (int i = ii; i < iend; i++) {
                    double* crow = C + (size_t)i * n;
                    for (int k = kk; k < kend; k++) {
                        double aik = A[(size_t)i * n + k];
                        const double* brow = B + (size_t)k * n;
                        for (int j = jj; j < jend; j++)
                            crow[j] += aik * brow[j];
                    }
                }
            }
        }
    }
}

/* Транспонування B один раз; потім dot-product рядок×рядок. */
static void mm_transposed(const double* A, const double* B, double* C, int n) {
    double* Bt = (double*)xalloc((size_t)n * n * sizeof(double), 64);
    if (!Bt) return;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * n + i] = B[(size_t)i * n + j];

    for (int i = 0; i < n; i++) {
        const double* arow = A + (size_t)i * n;
        double* crow = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double* brow = Bt + (size_t)j * n;
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += arow[k] * brow[k];
            crow[j] = s;
        }
    }
    xfree(Bt);
}

static double matmul_checksum(const double* C, size_t count) {
    double s = 0.0;
    for (size_t i = 0; i < count; i++) s += C[i];
    return s;
}

typedef struct {
    const char* name;
    mm_fn fn;
} mm_variant_t;

static void section_matmul(const cfg_t* cfg) {
    int n = cfg->matmul_n;
    size_t count = (size_t)n * n;

    out_printf("\n# == 3) Matrix multiplication (n=%d, %.1f MB per matrix) ==\n",
               n, (double)count * sizeof(double) / (1024.0 * 1024.0));
    out_printf("# variant, n, best_sec, GFLOPs, checksum\n");

    double* A = (double*)xalloc(count * sizeof(double), 64);
    double* B = (double*)xalloc(count * sizeof(double), 64);
    double* C = (double*)xalloc(count * sizeof(double), 64);
    if (!A || !B || !C) {
        out_printf("# matmul: alloc failed\n");
        xfree(A); xfree(B); xfree(C);
        return;
    }
    for (size_t i = 0; i < count; i++) { A[i] = rng_double(); B[i] = rng_double(); }

    mm_variant_t vars[] = {
        {"naive_ijk", mm_ijk},
        {"ikj", mm_ikj},
        {"tiled64", mm_tiled},
        {"transposed", mm_transposed},
    };
    int nv = (int)(sizeof(vars) / sizeof(vars[0]));

    for (int v = 0; v < nv; v++) {
        double best = 1e300, checksum = 0.0;
        for (int r = 0; r < cfg->matmul_reps; r++) {
            memset(C, 0, count * sizeof(double));
            double t0 = now_sec();
            vars[v].fn(A, B, C, n);
            double t1 = now_sec();
            double dt = t1 - t0;
            if (dt < best) { best = dt; checksum = matmul_checksum(C, count); }
        }
        double gflops = 2.0 * (double)n * n * n / best / 1e9;
        out_printf("MATMUL,%s,%d,%.4f,%.3f,%.6f\n",
                   vars[v].name, n, best, gflops, checksum);
    }

    xfree(A); xfree(B); xfree(C);
}

/* ================================================================== */
/* 4. False sharing                                                  */
/* ================================================================== */
typedef struct {
    _Alignas(CACHELINE) volatile uint64_t v;
    unsigned char pad[CACHELINE];
} fs_slot_t;

typedef struct {
    _Alignas(CACHELINE) _Atomic uint64_t v;
    unsigned char pad[CACHELINE];
} atomic_slot_t;

typedef struct {
    volatile uint64_t* target;
    uint64_t iters;
    barrier_t* bar;
} fs_arg_t;

static THREAD_FCN fs_worker(void* p) {
    fs_arg_t* a = (fs_arg_t*)p;
    barrier_wait(a->bar);
    volatile uint64_t* t = a->target;
    for (uint64_t i = 0; i < a->iters; i++) (*t)++;
    return 0;
}

typedef struct {
    double sec;
    uint64_t sum;
    uint64_t expected;
    int ok;
} share_result_t;

/* mode: 0 = unpadded (усі лічильники в одній лінії),
         1 = padded (кожен у своїй лінії),
         2 = single thread (базовий рівень) */
static int run_falseshare(int mode, int threads, uint64_t iters, int reps,
                          share_result_t* out) {
    uint64_t* flat = NULL;
    fs_slot_t* padded = NULL;
    if (mode == 0) {
        flat = (uint64_t*)xalloc((size_t)threads * sizeof(uint64_t) + CACHELINE, 64);
        if (!flat) return 0;
    } else if (mode == 1) {
        padded = (fs_slot_t*)xalloc((size_t)threads * sizeof(fs_slot_t), 64);
        if (!padded) return 0;
    }

    double best = 1e300;
    uint64_t best_sum = 0;

    for (int r = 0; r < reps; r++) {
        if (flat) memset(flat, 0, (size_t)threads * sizeof(uint64_t));
        if (padded) memset(padded, 0, (size_t)threads * sizeof(fs_slot_t));

        if (mode == 2) {
            /* один потік виконує всю роботу — масштабований baseline */
            fs_slot_t s;
            memset(&s, 0, sizeof(s));
            double t0 = now_sec();
            for (uint64_t i = 0; i < iters * (uint64_t)threads; i++) s.v++;
            double t1 = now_sec();
            if (t1 - t0 < best) { best = t1 - t0; best_sum = s.v; }
            continue;
        }

        barrier_t bar;
        barrier_init(&bar, threads);
        fs_arg_t* args = (fs_arg_t*)calloc((size_t)threads, sizeof(fs_arg_t));
        th_t* tids = (th_t*)calloc((size_t)threads, sizeof(th_t));
        if (!args || !tids) { free(args); free(tids); continue; }

        for (int t = 0; t < threads; t++) {
            args[t].iters = iters;
            args[t].bar = &bar;
            args[t].target = (mode == 0) ? &flat[t] : &padded[t].v;
            tids[t] = thread_start(fs_worker, &args[t]);
        }
        double t0 = now_sec();
        barrier_release(&bar);
        for (int t = 0; t < threads; t++) thread_join(tids[t]);
        double t1 = now_sec();

        uint64_t sum = 0;
        for (int t = 0; t < threads; t++)
            sum += (mode == 0) ? flat[t] : padded[t].v;
        if (t1 - t0 < best) { best = t1 - t0; best_sum = sum; }
        free(args);
        free(tids);
    }

    out->sec = best;
    out->sum = best_sum;
    out->expected = iters * (uint64_t)threads;
    out->ok = (best_sum == out->expected);
    xfree(flat);
    xfree(padded);
    return 1;
}

static void section_falseshare(const cfg_t* cfg) {
    out_printf("\n# == 4) False sharing: %d threads x %llu increments ==\n",
               cfg->threads, (unsigned long long)cfg->fs_iters);
    out_printf("# mode, threads, total_ops, sec, Mops, sum, expected, ok\n");

    const char* names[] = {"unpadded", "padded", "single"};
    for (int m = 0; m < 3; m++) {
        share_result_t r;
        if (!run_falseshare(m, cfg->threads, cfg->fs_iters, 3, &r)) {
            out_printf("# false sharing mode %s failed\n", names[m]);
            continue;
        }
        double total = (double)cfg->fs_iters * (m == 2 ? 1.0 : cfg->threads) *
                       (m == 2 ? (double)cfg->threads : 1.0);
        double mops = total / r.sec / 1e6;
        out_printf("FALSESHARE,%s,%d,%.0f,%.4f,%.2f,%llu,%llu,%s\n",
                   names[m], m == 2 ? 1 : cfg->threads, total, r.sec, mops,
                   (unsigned long long)r.sum, (unsigned long long)r.expected,
                   r.ok ? "OK" : "FAIL");
    }
}

/* ================================================================== */
/* 5. Перегони: volatile / atomic / mutex                            */
/* ================================================================== */
static volatile uint64_t g_race_counter __attribute__((aligned(64))) = 0;
static _Atomic uint64_t g_atomic_shared __attribute__((aligned(64))) = 0;
static uint64_t g_mutex_value __attribute__((aligned(64))) = 0;
static mutex_t g_mutex;

typedef enum {
    MODE_VOLATILE_SHARED = 0,
    MODE_ATOMIC_SHARED = 1,
    MODE_MUTEX_SHARED = 2,
    MODE_ATOMIC_PADDED = 3
} race_mode_t;

typedef struct {
    int tid;
    uint64_t iters;
    race_mode_t mode;
    atomic_slot_t* slots;
    barrier_t* bar;
} race_arg_t;

static THREAD_FCN race_worker(void* p) {
    race_arg_t* a = (race_arg_t*)p;
    barrier_wait(a->bar);
    switch (a->mode) {
    case MODE_VOLATILE_SHARED:
        for (uint64_t i = 0; i < a->iters; i++) g_race_counter++;
        break;
    case MODE_ATOMIC_SHARED:
        for (uint64_t i = 0; i < a->iters; i++)
            atomic_fetch_add_explicit(&g_atomic_shared, 1, memory_order_relaxed);
        break;
    case MODE_MUTEX_SHARED:
        for (uint64_t i = 0; i < a->iters; i++) {
            mutex_lock(&g_mutex);
            g_mutex_value++;
            mutex_unlock(&g_mutex);
        }
        break;
    case MODE_ATOMIC_PADDED:
        for (uint64_t i = 0; i < a->iters; i++)
            atomic_fetch_add_explicit(&a->slots[a->tid].v, 1, memory_order_relaxed);
        break;
    }
    return 0;
}

typedef struct {
    double sec;
    uint64_t value;
    uint64_t expected;
    int ok;
} race_result_t;

static int run_race(race_mode_t mode, int threads, uint64_t iters, int reps,
                    race_result_t* out) {
    atomic_slot_t* slots = NULL;
    if (mode == MODE_ATOMIC_PADDED) {
        slots = (atomic_slot_t*)xalloc((size_t)threads * sizeof(atomic_slot_t), 64);
        if (!slots) return 0;
    }

    double best = 1e300;
    uint64_t best_val = 0;

    for (int r = 0; r < reps; r++) {
        g_race_counter = 0;
        atomic_store(&g_atomic_shared, 0);
        g_mutex_value = 0;
        if (slots) {
            for (int t = 0; t < threads; t++)
                atomic_store(&slots[t].v, 0);
        }

        barrier_t bar;
        barrier_init(&bar, threads);
        race_arg_t* args = (race_arg_t*)calloc((size_t)threads, sizeof(race_arg_t));
        th_t* tids = (th_t*)calloc((size_t)threads, sizeof(th_t));
        if (!args || !tids) { free(args); free(tids); continue; }

        for (int t = 0; t < threads; t++) {
            args[t].tid = t;
            args[t].iters = iters;
            args[t].mode = mode;
            args[t].slots = slots;
            args[t].bar = &bar;
            tids[t] = thread_start(race_worker, &args[t]);
        }
        double t0 = now_sec();
        barrier_release(&bar);
        for (int t = 0; t < threads; t++) thread_join(tids[t]);
        double t1 = now_sec();

        uint64_t val = 0;
        switch (mode) {
        case MODE_VOLATILE_SHARED: val = g_race_counter; break;
        case MODE_ATOMIC_SHARED:   val = atomic_load(&g_atomic_shared); break;
        case MODE_MUTEX_SHARED:    val = g_mutex_value; break;
        case MODE_ATOMIC_PADDED:
            for (int t = 0; t < threads; t++)
                val += atomic_load(&slots[t].v);
            break;
        }
        if (t1 - t0 < best) { best = t1 - t0; best_val = val; }
        free(args);
        free(tids);
    }

    out->sec = best;
    out->value = best_val;
    out->expected = iters * (uint64_t)threads;
    out->ok = (best_val == out->expected);
    xfree(slots);
    return 1;
}

static void section_race(const cfg_t* cfg) {
    out_printf("\n# == 5) Contention: %d threads x %llu increments ==\n",
               cfg->threads, (unsigned long long)cfg->race_iters);
    out_printf("# mode, threads, total_ops, sec, Mops, value, expected, lost, ok\n");

    const char* names[] = {"volatile_shared", "atomic_shared",
                           "mutex_shared", "atomic_padded"};
    mutex_init(&g_mutex);
    for (int m = 0; m < 4; m++) {
        race_result_t r;
        if (!run_race((race_mode_t)m, cfg->threads, cfg->race_iters, 3, &r)) {
            out_printf("# race mode %s failed\n", names[m]);
            continue;
        }
        double total = (double)cfg->race_iters * cfg->threads;
        double mops = total / r.sec / 1e6;
        unsigned long long lost = (r.expected > r.value) ? (r.expected - r.value) : 0;
        out_printf("CONTENTION,%s,%d,%.0f,%.4f,%.2f,%llu,%llu,%llu,%s\n",
                   names[m], cfg->threads, total, r.sec, mops,
                   (unsigned long long)r.value, (unsigned long long)r.expected,
                   lost, r.ok ? "OK" : "LOST");
    }
    mutex_destroy(&g_mutex);
}

/* ================================================================== */
/* 6. Обхід матриці: рядки vs стовпці                               */
/* ================================================================== */
static uint64_t sum_rows(const uint64_t* a, int n) {
    uint64_t s = 0;
    for (int i = 0; i < n; i++) {
        const uint64_t* row = a + (size_t)i * n;
        for (int j = 0; j < n; j++) s += row[j];
    }
    return s;
}

static uint64_t sum_cols(const uint64_t* a, int n) {
    uint64_t s = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            s += a[(size_t)i * n + j];
    return s;
}

static void section_rows(const cfg_t* cfg) {
    int n = cfg->rows_n;
    size_t count = (size_t)n * n;
    uint64_t* a = (uint64_t*)xalloc(count * sizeof(uint64_t), 64);
    if (!a) { out_printf("# rows: alloc failed\n"); return; }
    for (size_t i = 0; i < count; i++) a[i] = rng_next() % 1000ULL;

    size_t bytes = count * sizeof(uint64_t);
    int reps = (int)((512ULL * 1024 * 1024) / bytes);
    if (reps < 1) reps = 1;
    if (reps > 100) reps = 100;

    out_printf("\n# == 6) Matrix traversal (n=%d, %.1f MB) ==\n",
               n, (double)bytes / (1024.0 * 1024.0));
    out_printf("# order, n, sec, GBps, checksum\n");

    volatile uint64_t sink = 0;
    double best_row = 1e300, best_col = 1e300;
    uint64_t sr = 0, sc = 0;

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        uint64_t s = sum_rows(a, n);
        double t1 = now_sec();
        if (t1 - t0 < best_row) { best_row = t1 - t0; sr = s; }
        sink += s;
    }
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        uint64_t s = sum_cols(a, n);
        double t1 = now_sec();
        if (t1 - t0 < best_col) { best_col = t1 - t0; sc = s; }
        sink += s;
    }

    out_printf("ROWS,row_major,%d,%.4f,%.3f,%llu\n", n, best_row,
               (double)bytes / best_row / 1e9, (unsigned long long)sr);
    out_printf("ROWS,column_major,%d,%.4f,%.3f,%llu\n", n, best_col,
               (double)bytes / best_col / 1e9, (unsigned long long)sc);
    out_printf("ROWS,equal,%d,%s,%llu\n", n, sr == sc ? "OK" : "DIFF",
               (unsigned long long)(sr > sc ? sr - sc : sc - sr));
    (void)sink;
    xfree(a);
}

/* ================================================================== */
/* 7. Самотестування                                                 */
/* ================================================================== */
static int g_fail = 0;

static void check(const char* name, int cond, const char* fmt, ...) {
    char detail[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    out_printf("TEST,%s,%s,%s\n", name, cond ? "PASS" : "FAIL", detail);
    if (!cond) g_fail = 1;
}

static void test_permutation(void) {
    size_t n = 100000;
    uint64_t* next = build_cycle(n);
    if (!next) { check("permutation", 0, "alloc failed"); return; }
    unsigned char* seen = (unsigned char*)calloc(n, 1);
    size_t idx = 0, count = 0;
    int bad = 0;
    for (size_t k = 0; k < n; k++) {
        if (idx >= n || seen[idx]) { bad = 1; break; }
        seen[idx] = 1;
        count++;
        idx = (size_t)next[idx];
    }
    check("permutation_cycle", !bad && count == n && idx == 0,
          "visited=%zu/%zu back_to_start=%d", count, n, idx == 0 ? 1 : 0);
    free(seen);
    xfree(next);
}

static double max_abs_diff(const double* a, const double* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void test_matmul(void) {
    int n = 129;
    size_t count = (size_t)n * n;
    double* A = (double*)xalloc(count * sizeof(double), 64);
    double* B = (double*)xalloc(count * sizeof(double), 64);
    double* C = (double*)xalloc(count * sizeof(double), 64);
    double* R = (double*)xalloc(count * sizeof(double), 64);
    if (!A || !B || !C || !R) {
        check("matmul_alloc", 0, "alloc failed");
        xfree(A); xfree(B); xfree(C); xfree(R);
        return;
    }
    for (size_t i = 0; i < count; i++) { A[i] = rng_double(); B[i] = rng_double(); }
    memset(R, 0, count * sizeof(double));
    mm_ijk(A, B, R, n);
    double scale = 0.0;
    for (size_t i = 0; i < count; i++) { double v = fabs(R[i]); if (v > scale) scale = v; }

    mm_variant_t vars[] = {
        {"naive_ijk", mm_ijk}, {"ikj", mm_ikj},
        {"tiled64", mm_tiled}, {"transposed", mm_transposed},
    };
    for (int v = 0; v < 4; v++) {
        memset(C, 0, count * sizeof(double));
        vars[v].fn(A, B, C, n);
        double rel = scale > 0 ? max_abs_diff(C, R, count) / scale : 0.0;
        char name[64];
        snprintf(name, sizeof(name), "matmul_%s", vars[v].name);
        check(name, rel < 1e-10, "max_rel_err=%.3e (n=%d)", rel, n);
    }
    xfree(A); xfree(B); xfree(C); xfree(R);
}

static void test_rows(void) {
    int n = 512;
    size_t count = (size_t)n * n;
    uint64_t* a = (uint64_t*)xalloc(count * sizeof(uint64_t), 64);
    if (!a) { check("rows_alloc", 0, "alloc failed"); return; }
    for (size_t i = 0; i < count; i++) a[i] = rng_next() % 1000ULL;
    uint64_t sr = sum_rows(a, n);
    uint64_t sc = sum_cols(a, n);
    check("rows_equals_cols", sr == sc, "row=%llu col=%llu diff=%llu",
          (unsigned long long)sr, (unsigned long long)sc,
          (unsigned long long)(sr > sc ? sr - sc : sc - sr));
    xfree(a);
}

static void test_falseshare(void) {
    share_result_t r;
    if (!run_falseshare(0, 4, 200000, 1, &r)) {
        check("falseshare_unpadded", 0, "run failed"); return;
    }
    check("falseshare_unpadded", r.ok, "sum=%llu expected=%llu",
          (unsigned long long)r.sum, (unsigned long long)r.expected);
    if (!run_falseshare(1, 4, 200000, 1, &r)) {
        check("falseshare_padded", 0, "run failed"); return;
    }
    check("falseshare_padded", r.ok, "sum=%llu expected=%llu",
          (unsigned long long)r.sum, (unsigned long long)r.expected);
}

static void test_race(void) {
    mutex_init(&g_mutex);
    race_result_t r;
    if (run_race(MODE_ATOMIC_SHARED, 4, 200000, 1, &r))
        check("race_atomic_shared", r.ok, "value=%llu expected=%llu",
              (unsigned long long)r.value, (unsigned long long)r.expected);
    else
        check("race_atomic_shared", 0, "run failed");

    if (run_race(MODE_MUTEX_SHARED, 4, 200000, 1, &r))
        check("race_mutex_shared", r.ok, "value=%llu expected=%llu",
              (unsigned long long)r.value, (unsigned long long)r.expected);
    else
        check("race_mutex_shared", 0, "run failed");

    if (run_race(MODE_ATOMIC_PADDED, 4, 200000, 1, &r))
        check("race_atomic_padded", r.ok, "value=%llu expected=%llu",
              (unsigned long long)r.value, (unsigned long long)r.expected);
    else
        check("race_atomic_padded", 0, "run failed");

    /* volatile не є синхронізацією: оновлення губляться, але їх не може
       бути більше за очікувану кількість */
    if (run_race(MODE_VOLATILE_SHARED, 4, 200000, 1, &r))
        check("race_volatile_loses", r.value <= r.expected,
              "value=%llu expected=%llu lost=%llu",
              (unsigned long long)r.value, (unsigned long long)r.expected,
              (unsigned long long)(r.expected - r.value));
    else
        check("race_volatile_loses", 0, "run failed");

    mutex_destroy(&g_mutex);
}

static int run_selftest(void) {
    out_printf("CACHE_BENCH SELFTEST\n");
    test_permutation();
    test_matmul();
    test_rows();
    test_falseshare();
    test_race();
    out_printf("SELFTEST,%s\n", g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}

/* ================================================================== */
/* main                                                              */
/* ================================================================== */
static void ensure_results_dir(void) {
#if defined(_WIN32)
    _mkdir("results");
#else
    mkdir("results", 0777);
#endif
}

/* Розмір кеш-лінії L1D, який повідомляє ОС (для перевірки оцінки). */
static int os_cacheline(void) {
#if defined(_WIN32)
    DWORD len = 0;
    if (!GetLogicalProcessorInformationEx(RelationCache, NULL, &len) || len == 0)
        return CACHELINE;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* buf =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)malloc(len);
    if (!buf) return CACHELINE;
    if (!GetLogicalProcessorInformationEx(RelationCache, buf, &len)) {
        free(buf);
        return CACHELINE;
    }
    int line = CACHELINE;
    for (DWORD off = 0; off < len; ) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* p =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)((char*)buf + off);
        if (p->Relationship == RelationCache &&
            p->Cache.Level == 1 && p->Cache.Type == CacheData &&
            p->Cache.LineSize > 0) {
            line = (int)p->Cache.LineSize;
            break;
        }
        off += p->Size;
    }
    free(buf);
    return line;
#else
  #ifdef _SC_LEVEL1_DCACHE_LINESIZE
    long v = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (v > 0) return (int)v;
  #endif
    return CACHELINE;
#endif
}

static void banner(const cfg_t* cfg) {
    out_printf("# CACHE / MEMORY / RACE BENCHMARK (C11)\n");
#if defined(_WIN32)
    out_printf("# platform: Windows / MinGW-w64 GCC\n");
#else
    out_printf("# platform: POSIX / GCC\n");
#endif
    out_printf("# threads=%d matmul_n=%d rows_n=%d max_mb=%d stride_mb=%d\n",
               cfg->threads, cfg->matmul_n, cfg->rows_n, cfg->max_mb, cfg->stride_mb);
    out_printf("INFO,threads,%d\n", cfg->threads);
    out_printf("INFO,matmul_n,%d\n", cfg->matmul_n);
    out_printf("INFO,rows_n,%d\n", cfg->rows_n);
    out_printf("INFO,cacheline,%d\n", CACHELINE);
    out_printf("INFO,cacheline_os,%d\n", os_cacheline());
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--selftest") == 0)
            return run_selftest();
    }

    const char* out_path = (argc > 1) ? argv[1] : "results/lab3.csv";
    const char* section = (argc > 2) ? argv[2] : "all";
    cfg_t cfg = load_cfg();

    ensure_results_dir();
    g_out = fopen(out_path, "wb");
    if (g_out) {
        /* UTF-8 BOM для коректного відкриття в Excel */
        fwrite("\xEF\xBB\xBF", 1, 3, g_out);
    } else {
        fprintf(stderr, "warning: cannot open %s, writing to stdout only\n", out_path);
    }

    banner(&cfg);

    int all = (strcmp(section, "all") == 0);
    if (all || strcmp(section, "lat") == 0)    section_latency(&cfg);
    if (all || strcmp(section, "stride") == 0) section_stride(&cfg);
    if (all || strcmp(section, "matmul") == 0) section_matmul(&cfg);
    if (all || strcmp(section, "share") == 0)  section_falseshare(&cfg);
    if (all || strcmp(section, "race") == 0)   section_race(&cfg);
    if (all || strcmp(section, "rows") == 0)   section_rows(&cfg);

    if (g_out) fclose(g_out);
    return 0;
}
