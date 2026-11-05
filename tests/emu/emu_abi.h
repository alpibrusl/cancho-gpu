/* What libcuda_emu.c and a compiled kernel (cuda_emu.h) share. */
#ifndef EMU_ABI_H
#define EMU_ABI_H
#include <stdint.h>

/* In an anonymous MAP_SHARED page, so the forked child can report to its parent. */
typedef struct {
    int code;              /* 0 while fine; else why the launch failed (libcuda_emu.c maps it) */
    unsigned blk_x, blk_y; /* where */
    unsigned tid;
    uint64_t fault_addr;   /* for a signal: si_addr */
    char msg[512];
} EmuInfo;

typedef struct {
    unsigned gx, gy, bx;
    uint64_t seed;
    int shuffle;           /* schedule 1: shuffled block and thread order */
    int poison;            /* the byte shared memory holds at the start of each block */
    EmuInfo *info;
} EmuLaunch;
#endif
