"""The emulating driver and NVRTC, called from Python with ctypes
(tests/cuda_emu_selftest.py, tests/cuda_emu_big.py)."""
import ctypes
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EMU = ROOT / "build" / "emu"
os.environ.setdefault("CANCHOGPU_EMU_CACHE", str(EMU / "cache"))
cu = ctypes.CDLL(str(EMU / "libcuda_emu.so"))
rtc = ctypes.CDLL(str(EMU / "libnvrtc_emu.so"))
P = ctypes.c_void_p


class Result:
    def __init__(self, rc, err, stderr):
        self.rc, self.err, self.stderr = rc, err, stderr


def compile_kernel(src):
    prog = P()
    rtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"kernel.cu", 0, None, None)
    opts = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_89")
    rc = rtc.nvrtcCompileProgram(prog, 1, opts)
    n = ctypes.c_size_t()
    rtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value + 1)
    rtc.nvrtcGetProgramLog(prog, log)
    so = None
    if rc == 0:
        rtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        rtc.nvrtcGetPTX(prog, buf)
        so = buf.value
    return rc, log.value.decode(), so


def run(src, entry, grid, block, inputs, passes=2, seed=1):
    """inputs: list of bytes (one buffer each). Returns (Result, [bytes out])."""
    rc, log, so = compile_kernel(src)
    assert rc == 0, log
    mod, fn = P(), P()
    assert cu.cuModuleLoadData(ctypes.byref(mod), so) == 0
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, entry.encode()) == 0
    ptrs = []
    for data in inputs:
        d = ctypes.c_ulonglong()
        assert cu.cuMemAlloc_v2(ctypes.byref(d), len(data)) == 0
        assert cu.cuMemcpyHtoD_v2(d, data, len(data)) == 0
        ptrs.append(d)
    params = (P * 64)(*[ctypes.cast(ctypes.pointer(d), P) for d in ptrs])
    os.environ["CANCHOGPU_EMU_PASSES"] = str(passes)
    os.environ["CANCHOGPU_EMU_SEED"] = str(seed)
    with tempfile.TemporaryFile() as t:
        saved = os.dup(2)
        os.dup2(t.fileno(), 2)
        try:
            rc = cu.cuLaunchKernel(fn, grid[0], grid[1], 1, block, 1, 1, 0, None, params, None)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        t.seek(0)
        stderr = t.read().decode()
    s = ctypes.c_char_p()
    cu.cuGetErrorString(rc, ctypes.byref(s))
    outs = []
    for data, d in zip(inputs, ptrs):
        out = ctypes.create_string_buffer(len(data))
        cu.cuMemcpyDtoH_v2(out, d, len(data))
        outs.append(out.raw)
    for d in ptrs:
        cu.cuMemFree_v2(d)
    return Result(rc, (s.value or b"").decode() if rc else "", stderr), outs


