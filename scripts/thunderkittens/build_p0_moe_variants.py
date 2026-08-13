#!/usr/bin/env python3
"""Build the three P0 MoE _C variants outside the accel-sim checkout."""

import argparse
import os
import pathlib
import subprocess
import sys
import sysconfig

from p0_manifest import load_manifest, variant_name


def specialized_source(original, shape):
    replacements = {
        "static constexpr int H = 7168;":
            "static constexpr int H = {};".format(shape["H"]),
        "static constexpr int I = 2048;":
            "static constexpr int I = {};".format(shape["I"]),
        "static constexpr int TOP_K = 8;":
            "static constexpr int TOP_K = {};".format(shape["top_k"]),
        "static constexpr int NUM_EXPERTS = 256;":
            "static constexpr int NUM_EXPERTS = {};".format(shape["experts"]),
    }
    result = original
    for before, after in replacements.items():
        if result.count(before) != 1:
            raise ValueError("source constant not found exactly once: {}".format(before))
        result = result.replace(before, after)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--group", default="p0")
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _, cases = load_manifest(args.manifest, args.group)
    shapes = {variant_name(c["execution"]): c["execution"]
              for c in cases.values() if c["family"] == "MoE"}
    if len(shapes) != 3:
        parser.error("P0 must resolve to exactly three MoE variants")
    tk_root = pathlib.Path(os.environ.get(
        "TK_ROOT", pathlib.Path.home() / "Repos/ThunderKittens")).resolve()
    source_path = tk_root / "kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu"
    common_makefile = tk_root / "kernels/common.mk"
    original = source_path.read_text(encoding="utf-8")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    for name, shape in sorted(shapes.items()):
        directory = args.output_root.resolve() / name
        generated = directory / "moe_dispatch_gemm_h100.cu"
        output = directory / "_C{}".format(suffix)
        print("MOE_VARIANT name={} H={} I={} top_k={} experts={} output={}".format(
            name, shape["H"], shape["I"], shape["top_k"], shape["experts"],
            output), flush=True)
        if args.dry_run or output.is_file():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        generated.write_text(specialized_source(original, shape), encoding="utf-8")
        env = os.environ.copy()
        env["PATH"] = "{}:{}".format(pathlib.Path(sys.executable).resolve().parent,
                                      env.get("PATH", ""))
        subprocess.run([
            "make", "-f", str(common_makefile), "ARCH=SM90", "CONFIG=pytorch",
            "CMD=true", "SRC={}".format(generated), "OUT={}".format(output),
        ], cwd=tk_root / "kernels", env=env, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
