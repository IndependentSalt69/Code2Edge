#!/usr/bin/env python3
"""
Build script to compile src/pipeline/feature_extraction.c into a native shared library for host parity tests.
Supports MSVC cl.exe on Windows as well as gcc / clang on POSIX systems.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"


def get_msvc_cl() -> Path | None:
    cl_candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"),
    ]
    for c in cl_candidates:
        if c.is_file():
            return c
        parent = c.parent.parent.parent
        if parent.exists():
            for cl in parent.glob("*/bin/Hostx64/x64/cl.exe"):
                if cl.is_file():
                    return cl
    return None


def compile_c_dll(force: bool = False) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    system = platform.system().lower()

    if system == "windows":
        dll_name = "feature_extraction.dll"
    elif system == "darwin":
        dll_name = "libfeature_extraction.dylib"
    else:
        dll_name = "libfeature_extraction.so"

    dll_path = BUILD_DIR / dll_name

    c_src = ROOT / "src" / "pipeline" / "feature_extraction.c"
    runtime_src = ROOT / "tools" / "host_compat" / "host_runtime.c"
    include_pipeline = ROOT / "src" / "pipeline"
    include_compat = ROOT / "tools" / "host_compat"

    if dll_path.exists() and not force:
        # Check timestamps
        dll_mtime = dll_path.stat().st_mtime
        if dll_mtime >= c_src.stat().st_mtime and dll_mtime >= runtime_src.stat().st_mtime:
            return dll_path

    if system == "windows":
        cl_exe = get_msvc_cl()
        if not cl_exe:
            raise RuntimeError("MSVC cl.exe not found on system.")

        cmd = [
            str(cl_exe),
            "/LD",
            "/O2",
            "/Oi-",
            "/Gs9999999",
            f"/I{include_compat}",
            f"/I{include_pipeline}",
            str(c_src),
            str(runtime_src),
            f"/Fe:{dll_path}",
            f"/Fo:{BUILD_DIR}\\",
            "/link",
            "/NODEFAULTLIB",
            "/NOENTRY",
            "/STACK:4194304",
            "/EXPORT:feature_extraction_init",
            "/EXPORT:feature_extraction_run",
            "/EXPORT:feature_extraction_run_f32",
            "/EXPORT:feature_extraction_stages_f32",
            "/EXPORT:feature_extraction_compute_checksum",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("MSVC stdout:\n", res.stdout, file=sys.stderr)
            print("MSVC stderr:\n", res.stderr, file=sys.stderr)
            raise RuntimeError(f"cl.exe build failed with exit code {res.returncode}")
    else:
        cmd = [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            f"-I{include_pipeline}",
            str(c_src),
            "-o",
            str(dll_path),
            "-lm",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"gcc build failed: {res.stderr}")

    return dll_path


if __name__ == "__main__":
    out = compile_c_dll(force=True)
    print(f"Compiled C pipeline shared library: {out}")
