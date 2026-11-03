"""transformers' qwen3_5 modeling code, read from the installed package and recompiled with its float32 casts turned into float64, so that torch autograd
through HF's own code can be the reference for a binary64 specification (the shipped code rounds to float32 inside the delta rule, the gates and the norms)."""
import importlib
import inspect
import sys
import types


def load():
    import transformers.models.qwen3_5.modeling_qwen3_5 as orig
    src = inspect.getsource(orig)
    src = src.replace("torch.float32", "torch.float64").replace(".float()", ".double()")
    name = "transformers.models.qwen3_5.modeling_qwen3_5_f64"
    mod = types.ModuleType(name)
    mod.__package__ = "transformers.models.qwen3_5"
    mod.__file__ = orig.__file__
    sys.modules[name] = mod
    exec(compile(src, orig.__file__ + " [float64]", "exec"), mod.__dict__)
    return mod
