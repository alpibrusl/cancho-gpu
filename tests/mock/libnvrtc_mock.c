/* A stand-in for NVRTC (docs/device.md section 5): "compiling" yields the
 * source itself as the PTX, so the mock driver can see which entries it
 * defines. Records the options it was given in $LEXGPU_MOCK_LOG. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int nvrtcResult;
struct prog { char *src; };

nvrtcResult nvrtcCreateProgram(void **p, const char *src, const char *name, int n, const char *const *h,
                               const char *const *i) {
    (void)name; (void)n; (void)h; (void)i;
    struct prog *x = malloc(sizeof *x);
    x->src = strdup(src);
    *p = x;
    return 0;
}
nvrtcResult nvrtcCompileProgram(void *p, int n, const char *const *opts) {
    (void)p;
    const char *path = getenv("LEXGPU_MOCK_LOG");
    FILE *log = path ? fopen(path, "a") : NULL;
    if (log) {
        fprintf(log, "compile");
        for (int i = 0; i < n; i++) {
            fprintf(log, " %s", opts[i]);
        }
        fprintf(log, "\n");
        fclose(log);
    }
    return 0;
}
nvrtcResult nvrtcGetPTXSize(void *p, size_t *n) { *n = strlen(((struct prog *)p)->src); return 0; }
nvrtcResult nvrtcGetPTX(void *p, char *out) { memcpy(out, ((struct prog *)p)->src, strlen(((struct prog *)p)->src)); return 0; }
nvrtcResult nvrtcGetProgramLogSize(void *p, size_t *n) { (void)p; *n = 0; return 0; }
nvrtcResult nvrtcGetProgramLog(void *p, char *out) { (void)p; (void)out; return 0; }
nvrtcResult nvrtcDestroyProgram(void **p) { struct prog *x = *p; free(x->src); free(x); return 0; }
