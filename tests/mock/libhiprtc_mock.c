/* A stand-in for hipRTC (docs/amd.md section 3.5): "compiling" yields the source itself as the code, so the mock runtime can see
 * which entries it defines. Records the program's name and the options it was given in $CANCHOGPU_MOCK_LOG, as
 * `compile-hip <name> <options...>`. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int hiprtcResult;
struct prog { char *src; char *name; };

hiprtcResult hiprtcCreateProgram(void **p, const char *src, const char *name, int n, const char **h, const char **i) {
    (void)n; (void)h; (void)i;
    struct prog *x = malloc(sizeof *x);
    x->src = strdup(src);
    x->name = strdup(name);
    *p = x;
    return 0;
}
hiprtcResult hiprtcCompileProgram(void *p, int n, const char **opts) {
    const char *path = getenv("CANCHOGPU_MOCK_LOG");
    FILE *log = path ? fopen(path, "a") : NULL;
    if (log) {
        fprintf(log, "compile-hip %s", ((struct prog *)p)->name);
        for (int i = 0; i < n; i++) {
            fprintf(log, " %s", opts[i]);
        }
        fprintf(log, "\n");
        fclose(log);
    }
    return 0;
}
hiprtcResult hiprtcGetCodeSize(void *p, size_t *n) { *n = strlen(((struct prog *)p)->src); return 0; }
hiprtcResult hiprtcGetCode(void *p, char *out) { memcpy(out, ((struct prog *)p)->src, strlen(((struct prog *)p)->src)); return 0; }
hiprtcResult hiprtcGetProgramLogSize(void *p, size_t *n) { (void)p; *n = 0; return 0; }
hiprtcResult hiprtcGetProgramLog(void *p, char *out) { (void)p; (void)out; return 0; }
hiprtcResult hiprtcDestroyProgram(void **p) { struct prog *x = *p; free(x->src); free(x->name); free(x); return 0; }
