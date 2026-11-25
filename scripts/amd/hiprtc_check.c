/* Give kernel sources to a hipRTC and say which compile (docs/amd.md slice H6).
 *
 *     hiprtc_check <libhiprtc.so> [-o <option>]... <file.hip>...
 *
 * One line a file on standard output, `ok <file>` or `FAIL <file>`, the
 * compiler's log for a failure on standard error, and a count at the end.
 * Exit 1 when any file failed, 2 on a usage or loading error. The library is
 * whatever the first argument names: AMD's, or tests/emu/libhiprtc_emu.so. */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *need(void *lib, const char *name) {
    void *p = dlsym(lib, name);
    if (!p) {
        fprintf(stderr, "hiprtc_check: the library has no %s\n", name);
        exit(2);
    }
    return p;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = calloc(1, (size_t)n + 1);
    if (s && fread(s, 1, (size_t)n, f) != (size_t)n) {
        free(s);
        s = NULL;
    }
    fclose(f);
    return s;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: hiprtc_check <libhiprtc.so> [-o <option>]... <file.hip>...\n");
        return 2;
    }
    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "hiprtc_check: %s\n", dlerror());
        return 2;
    }
    int (*create)(void **, const char *, const char *, int, const char **, const char **) = need(lib, "hiprtcCreateProgram");
    int (*compile)(void *, int, const char **) = need(lib, "hiprtcCompileProgram");
    int (*logsize)(void *, size_t *) = need(lib, "hiprtcGetProgramLogSize");
    int (*getlog)(void *, char *) = need(lib, "hiprtcGetProgramLog");
    int (*destroy)(void **) = need(lib, "hiprtcDestroyProgram");
    const char *opts[32];
    int n_opts = 0, at = 2;
    while (at + 1 < argc && !strcmp(argv[at], "-o") && n_opts < 32) {
        opts[n_opts++] = argv[at + 1];
        at += 2;
    }
    int ok = 0, bad = 0;
    for (; at < argc; at++) {
        char *src = slurp(argv[at]);
        if (!src) {
            fprintf(stderr, "hiprtc_check: cannot read %s\n", argv[at]);
            return 2;
        }
        void *prog = NULL;
        int rc = create(&prog, src, "kernel.hip", 0, NULL, NULL);
        if (!rc) {
            rc = compile(prog, n_opts, opts);
        }
        if (rc) {
            size_t n = 0;
            char *log = NULL;
            if (prog && !logsize(prog, &n) && n > 0 && (log = calloc(1, n + 1))) {
                getlog(prog, log);
            }
            printf("FAIL %s\n", argv[at]);
            fprintf(stderr, "--- %s (hiprtc %d)\n%s\n", argv[at], rc, log ? log : "(no log)");
            free(log);
            bad++;
        } else {
            printf("ok %s\n", argv[at]);
            ok++;
        }
        if (prog) {
            destroy(&prog);
        }
        free(src);
        fflush(stdout);
    }
    printf("hiprtc: %d compiled, %d failed\n", ok, bad);
    return bad ? 1 : 0;
}
