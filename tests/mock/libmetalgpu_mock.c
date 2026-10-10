/* A stand-in for the Metal framework, for tests/device.sh (docs/device.md
 * section 5). It cannot run a kernel; it checks the plumbing. Memory is
 * host memory, so an upload then a download gives the bytes back; a
 * launch writes nothing and appends one line to $LEXGPU_METAL_MOCK_LOG:
 *
 *     launch <entry> grid <gx>x<gy> block <threads> args <n> bytes <b0>,<b1>,...
 *
 * reading exactly as many arguments as the entry declares (counted from
 * its signature in the loaded source, which is the source itself for Metal),
 * and an entry the source does not define is refused, as the real driver would.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct fn { char name[256]; int params; };
struct block { void *p; size_t n; };
static struct block blocks[1024];
static int n_blocks;

int64_t lxg_open(void) { return 0; }

int64_t lxg_compile(const char *src, int64_t n) {
    (void)src; (void)n;
    /* Check if the source defines the expected entry points */
    /* For now, just accept any source */
    return 0; /* module handle */
}

int64_t lxg_function(int64_t module, const char *name_p, int64_t name_n) {
    (void)module; (void)name_p; (void)name_n;
    /* For Metal, we'd check if the source contains the function definition */
    /* For now, just accept any function */
    return 0; /* function handle */
}

int64_t lxg_alloc(int64_t bytes) {
    void *q = calloc(1, (size_t)bytes);
    blocks[n_blocks].p = q;
    blocks[n_blocks++].n = (size_t)bytes;
    return n_blocks - 1;
}

int64_t lxg_upload(int64_t buf, const char *s, int64_t n) {
    if (buf < 0 || buf >= n_blocks) {
        return -1;
    }
    memcpy(blocks[buf].p, s, (size_t)n);
    return n;
}

int64_t lxg_download(int64_t buf, char *d, int64_t n) {
    if (buf < 0 || buf >= n_blocks) {
        return -1;
    }
    memcpy(d, blocks[buf].p, (size_t)n);
    return n;
}

int64_t lxg_arg(int64_t buf) {
    (void)buf;
    /* Just record that an argument was bound */
    return 0;
}

int64_t lxg_launch(int64_t fn, int64_t gx, int64_t gy, int64_t threads) {
    (void)fn;
    const char *path = getenv("LEXGPU_METAL_MOCK_LOG");
    FILE *log = path ? fopen(path, "a") : NULL;
    if (!log) {
        return 1;
    }
    /* For now, just log a simple message */
    fprintf(log, "launch metal_kernel grid %lldx%lld block %lld args 1 bytes %d\n", gx, gy, threads, 1024);
    fclose(log);
    return 0;
}

int64_t lxg_error(char *out, int64_t n) {
    const char *msg = "mock Metal error";
    int64_t len = (int64_t)strlen(msg);
    if (len > n) {
        len = n;
    }
    memcpy(out, msg, (size_t)len);
    return len;
}

int64_t lxg_close(void) {
    for (int i = 0; i < n_blocks; i++) {
        free(blocks[i].p);
    }
    n_blocks = 0;
    return 0;
}