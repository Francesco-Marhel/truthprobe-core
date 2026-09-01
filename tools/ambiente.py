# -*- coding: utf-8 -*-
"""
ambiente.py  --  check the machine BEFORE downloading twenty gigabytes.

Run this first on any machine that is not yours. It reports what is installed,
whether the GPU actually computes, and how much of a model would fit in float32,
which is the precision the identity gate requires.

WHY float32 IS THE BINDING CONSTRAINT
The additive decomposition subtracts large residual states to obtain small
per-block deltas. In bfloat16 the states themselves carry about three decimal
digits, so the subtraction loses most of what it was meant to measure and the
gate fails. Everything downstream of that gate is therefore float32, and the
memory that matters is TOTAL parameters times four bytes: for a mixture of
experts, the active count is irrelevant, because every expert must be resident.

THE BLACKWELL TRAP
A PyTorch wheel built for CUDA 12.1 or earlier contains no kernels for compute
capability 12.0, which is what an RTX 50-series card reports. It installs, it
imports, `cuda.is_available()` returns True, the device name prints correctly,
and then the first real operation fails. This tool does not ask the library
whether the GPU works: it runs a matrix multiplication and checks the result.

    python ambiente.py
    python ambiente.py --model allenai/OLMoE-1B-7B-0125
"""

import argparse
import platform
import shutil
import sys


def gb(x):
    return x / (1024 ** 3)


def check_python():
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) < (3, 14)
    print("  python %d.%d.%d   %s" % (v.major, v.minor, v.micro,
                                      "ok" if ok else "OUTSIDE 3.11-3.13"))
    if not ok:
        print("    below 3.11 some torch wheels are awkward to find; 3.13 is")
        print("    still uneven. 3.11 or 3.12 is the comfortable range.")
    return ok


def check_venv():
    inside = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print("  virtual environment: %s" % ("yes" if inside else "NO"))
    if not inside:
        print("    installing into the system Python on someone else's machine")
        print("    is how you leave a mess behind. python -m venv venv")
    return inside


def check_ram():
    try:
        import psutil
        tot = psutil.virtual_memory().total
        av = psutil.virtual_memory().available
        print("  system RAM: %.1f GB total, %.1f GB available" % (gb(tot), gb(av)))
        return gb(tot), gb(av)
    except ImportError:
        if platform.system() == "Windows":
            import ctypes

            class S(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            s = S()
            s.dwLength = ctypes.sizeof(S)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            print("  system RAM: %.1f GB total, %.1f GB available"
                  % (gb(s.ullTotalPhys), gb(s.ullAvailPhys)))
            return gb(s.ullTotalPhys), gb(s.ullAvailPhys)
        print("  system RAM: unknown (pip install psutil to see it)")
        return None, None


def check_torch():
    try:
        import torch
    except ImportError:
        print("  torch: NOT INSTALLED")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return None, None
    print("  torch %s   built for CUDA %s"
          % (torch.__version__, torch.version.cuda or "cpu only"))
    if not torch.cuda.is_available():
        print("  CUDA: not available. Everything will run on CPU, which is slow")
        print("    but correct: no result depends on the device.")
        return torch, None

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    tot = torch.cuda.get_device_properties(0).total_memory
    print("  GPU: %s   compute capability %d.%d   %.1f GB"
          % (name, cap[0], cap[1], gb(tot)))

    # DO NOT trust is_available(): run something and check the answer.
    try:
        a = torch.randn(64, 64, device="cuda")
        r = float((a @ a.T).sum())
        if r != r:
            raise RuntimeError("result is NaN")
        print("  GPU compute: ok (a real matrix multiplication ran)")
    except Exception as e:
        print("  GPU compute: FAILED  (%s)" % type(e).__name__)
        print("    %s" % str(e).splitlines()[0][:100])
        if cap[0] >= 12:
            print("    compute capability %d.%d is Blackwell. A wheel built for" % cap)
            print("    CUDA 12.1 or earlier has no kernels for it: it installs,")
            print("    it imports, it reports the card, and then it cannot compute.")
            print("    pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return torch, None
    return torch, gb(tot)


def check_libs():
    """Report versions WITHOUT importing the heavy packages.

    Importing transformers walks every model directory to build its import
    structure, which on a fresh environment takes many seconds with no output.
    A tool that looks frozen invites Ctrl+C, and the person then believes it
    crashed. importlib.metadata reads the installed version from disk instead,
    which is instant and answers the question actually being asked: is it
    installed, and which version."""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        from importlib_metadata import version, PackageNotFoundError

    out = {}
    for name, why in (("transformers", "loading models"),
                      ("datasets", "loading CounterFact"),
                      ("safetensors", "loading weights without pickle"),
                      ("truthprobe", "the library itself")):
        try:
            v = version(name)
            print("  %-14s %-10s ok" % (name, v))
            out[name] = v
        except PackageNotFoundError:
            print("  %-14s %-10s MISSING   (%s)" % (name, "", why))
            out[name] = None
    return out


def check_library_path():
    """Where truthprobe actually resolves from.

    Worth printing: an editable install records an ABSOLUTE path, so renaming
    or moving the folder leaves the installation pointing at somewhere that no
    longer exists. The cure is to reinstall from the new folder."""
    try:
        import truthprobe
        print("  truthprobe resolves to: %s" % truthprobe.__file__)
    except ImportError:
        print("  truthprobe: cannot be imported")
        print("    if the folder was renamed or moved, the editable install")
        print("    still points at the old path: rerun pip install -e . there")


def fit_table(ram, vram):
    """How much model fits, in float32, which is what the gate requires."""
    models = [("Gemma-4-E2B", 5.0), ("OLMoE-1B-7B", 6.9), ("OLMo-2-1B", 1.5),
              ("Gemma-4-E4B", 8.0), ("Qwen1.5-MoE-A2.7B", 14.3),
              ("DeepSeek-MoE-16B", 16.0)]
    budget_ram = (ram - 4.0) if ram else None      # 4 GB for Python and the rest
    print()
    print("  WHAT FITS IN float32 (total parameters x 4 bytes)")
    print("  %-22s %8s %10s %10s" % ("model", "total B", "fp32 GB", "verdict"))
    print("  " + "-" * 54)
    for name, b in models:
        need = b * 4
        if vram and need <= vram - 2:
            v = "GPU"
        elif budget_ram and need <= budget_ram:
            v = "RAM"
        elif budget_ram and vram and need <= budget_ram + vram - 2:
            v = "split"
        else:
            v = "no"
        print("  %-22s %8.1f %10.1f %10s" % (name, b, need, v))
    print()
    print("  GPU    fits on the card alone")
    print("  RAM    fits in system memory; use device_map='auto' with a VRAM cap")
    print("  split  needs both, weights moving each pass: correct but slow")
    print()
    print("  For a mixture of experts the ACTIVE parameter count is irrelevant:")
    print("  every expert must be resident, so read the total column.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default=None,
                    help="also report the size of this model on the Hub, "
                         "without downloading the weights")
    a = ap.parse_args()

    print()
    print("=" * 66)
    print("ENVIRONMENT CHECK")
    print("=" * 66)
    print("[system] %s %s" % (platform.system(), platform.machine()))
    check_python()
    check_venv()
    ram, _ = check_ram()
    free = shutil.disk_usage(".").free
    print("  free disk here: %.1f GB" % gb(free))
    print()
    print("[compute]")
    torch, vram = check_torch()
    print()
    print("[libraries]  (versions read from disk, nothing is imported)")
    libs = check_libs()
    check_library_path()

    if ram:
        fit_table(ram, vram)

    if a.model:
        print("[model] %s" % a.model)
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(a.model, files_metadata=True)
            tot = sum(f.size or 0 for f in info.siblings
                      if f.rfilename.endswith((".safetensors", ".bin")))
            print("  weights on the Hub: %.1f GB (as stored, usually bfloat16)"
                  % gb(tot))
            print("  in float32 that becomes roughly %.1f GB" % (gb(tot) * 2))
            if free < tot * 1.1:
                print("  NOT ENOUGH DISK: %.1f GB free, %.1f GB needed"
                      % (gb(free), gb(tot)))
        except Exception as e:
            print("  could not query the Hub (%s)" % type(e).__name__)
        print()

    missing = [k for k, v in libs.items() if v is None]
    print("=" * 66)
    if missing:
        print("MISSING: %s" % ", ".join(missing))
        print()
        print("  python -m venv venv")
        print("  venv\\Scripts\\Activate.ps1          # or source venv/bin/activate")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print("  pip install \"git+https://github.com/Francesco-Marhel/TruthProbe"
              "@COMMIT#egg=truthprobe[models]\"")
        print()
        print("  The [models] extra pulls transformers, datasets and safetensors.")
        print("  Nothing else needs installing by hand.")
    else:
        print("ready")
    print()


if __name__ == "__main__":
    main()
