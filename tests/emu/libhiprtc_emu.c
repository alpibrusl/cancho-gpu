/* A hipRTC that compiles for the CPU (docs/amd.md section 7): libnvrtc_emu.c
 * under HIP's names, with what differs taken from EMU_HIP -- a wave of 64
 * lanes unless CANCHOGPU_EMU_WARP says 32 (gfx10 and later), `hip/hip_fp16.h`
 * for the header, no wmma, and the "code" it hands back is the path of the
 * shared object, which libamdhip64_emu.c dlopens.
 *
 * It says nothing of AMD's compiler: the header it reads is this
 * repository's, not ROCm's. */
#define EMU_HIP 1
#include "libnvrtc_emu.c"

typedef int hiprtcResult;
hiprtcResult hiprtcCreateProgram(void **p, const char *src, const char *name, int n, const char **h, const char **i) {
    return nvrtcCreateProgram(p, src, name, n, (const char *const *)h, (const char *const *)i);
}
hiprtcResult hiprtcCompileProgram(void *p, int n, const char **opts) {
    return nvrtcCompileProgram(p, n, (const char *const *)opts);
}
hiprtcResult hiprtcGetCodeSize(void *p, size_t *n) { return nvrtcGetPTXSize(p, n); }
hiprtcResult hiprtcGetCode(void *p, char *out) { return nvrtcGetPTX(p, out); }
hiprtcResult hiprtcGetProgramLogSize(void *p, size_t *n) { return nvrtcGetProgramLogSize(p, n); }
hiprtcResult hiprtcGetProgramLog(void *p, char *out) { return nvrtcGetProgramLog(p, out); }
hiprtcResult hiprtcDestroyProgram(void **p) { return nvrtcDestroyProgram(p); }
