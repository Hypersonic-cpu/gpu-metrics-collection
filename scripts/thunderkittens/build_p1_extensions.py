#!/usr/bin/env python3
"""Build repo-local P1 extensions from copied TK sources; originals stay untouched."""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "p1_extensions"


def replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise RuntimeError("{} replacement expected once, found {}".format(
            label, text.count(old)))
    return text.replace(old, new, 1)


def write_if_changed(path, content):
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def make_mha_copy(original):
    original = replace_once(
        original, "#ifdef TORCH_COMPILE", "#if 1", "MHA PyTorch mode")
    start = original.index("std::vector<at::Tensor> \nattention_forward")
    end = original.index("std::vector<at::Tensor> \nattention_backward")
    prefix, forward, suffix = original[:start], original[start:end], original[end:]
    forward = replace_once(
        forward,
        "attention_forward(at::Tensor q, at::Tensor k, at::Tensor v, bool causal)",
        "attention_forward(at::Tensor q, at::Tensor k, at::Tensor v, "
        "at::Tensor o, at::Tensor l_vec, bool causal)",
        "forward signature")
    forward = forward.replace("seq_len", "q_len")
    forward = replace_once(
        forward, "auto q_len  = q.size(2);",
        "auto q_len  = q.size(2);\n    auto kv_len = k.size(2);",
        "q/kv lengths")
    forward = forward.replace(
        'TORCH_CHECK(k.size(2) == q_len, "K sequence length dimension - idx 2 - must match for all inputs");',
        'TORCH_CHECK(k.size(2) == kv_len, "K sequence length mismatch");')
    forward = forward.replace(
        'TORCH_CHECK(v.size(2) == q_len, "V sequence length dimension - idx 2 - must match for all inputs");',
        'TORCH_CHECK(v.size(2) == kv_len, "V sequence length mismatch");')
    allocation_start = forward.index("    // for the returned outputs")
    allocation_end = forward.index("    float* l_ptr")
    replacement = """    CHECK_INPUT(o);
    CHECK_INPUT(l_vec);
    TORCH_CHECK(o.sizes() == at::IntArrayRef({batch, qo_heads, q_len, head_dim}),
                "O shape mismatch");
    TORCH_CHECK(l_vec.sizes() == at::IntArrayRef({batch, qo_heads, q_len, 1}),
                "L shape mismatch");

    bf16* d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());

"""
    forward = forward[:allocation_start] + replacement + forward[allocation_end:]
    forward = forward.replace(
        "static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(q_len)",
        "static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kv_len)")
    forward = forward.replace(
        "            static_cast<int>(q_len),\n            static_cast<int>(hr)",
        "            static_cast<int>(kv_len),\n            static_cast<int>(hr)")
    combined = prefix + forward + suffix
    combined = replace_once(
        combined, 'm.def("mha_forward",  attention_forward,',
        'm.def("mha_forward_into", attention_forward,', "pybind name")
    return combined


def make_gemm_copy(original):
    original = replace_once(
        original, "#include <iostream>",
        "#include <iostream>\n#include <chrono>\n#include <filesystem>\n"
        "#include <thread>\n#include <cstdlib>", "GEMM gate headers")
    text = replace_once(
        original,
        "double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false)",
        "double run_benchmark(size_t M, size_t N, size_t K, "
        "int rotation_buffers, int warmup_rounds, int num_iters, bool check, "
        "const std::string &case_id)",
        "GEMM benchmark signature")
    start = text.index("    // L2 cache eviction - multiple buffer groups")
    end = text.index("    // Allocate device memory", start)
    text = text[:start] + (
        "    const int arg_group_count = rotation_buffers;\n"
        "    const size_t arg_size = 2 * (size_t(M) * K + size_t(N) * K + "
        "size_t(M) * N);\n\n") + text[end:]
    old_iters = """    // Number of iterations
    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;
"""
    text = replace_once(
        text, old_iters,
        "    const int num_warmups = warmup_rounds * arg_group_count;\n",
        "GEMM iteration policy")
    warmup_end = """    for(int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        inner_run<mmt>(d_A[idx], d_B[idx], d_C[idx], M, N, K, grid, block);
    }
"""
    text = replace_once(
        text, warmup_end, warmup_end +
        "    CUDACHECK(cudaDeviceSynchronize());\n"
        "    std::cout << \"ROTATION_ALLOC_DONE case=\" << case_id "
        "<< \" copies=\" << arg_group_count "
        "<< \" working_set_bytes_per_gpu=\" << arg_size * arg_group_count "
        "<< \" seed_base=42 seed_stride=100 fingerprint_first=seed42_43 \" "
        "<< \"fingerprint_last=seed\" << 42 + (arg_group_count-1)*100 "
        "<< \"_\" << 43 + (arg_group_count-1)*100 << \"\\n\";\n"
        "    std::cout << \"ROTATION_WARMUP_DONE case=\" << case_id "
        "<< \" rounds=\" << warmup_rounds "
        "<< \" kernels=\" << num_warmups << \"\\n\";\n"
        "    std::cout << \"ROTATION_PROFILE_READY case=\" << case_id "
        "<< std::endl;\n"
        "    if (const char *start_file = std::getenv(\"ROTATION_START_FILE\")) {\n"
        "        while (!std::filesystem::exists(start_file))\n"
        "            std::this_thread::sleep_for(std::chrono::milliseconds(1));\n"
        "    }\n"
        "    std::cout << \"ROTATION_MEASURE_BEGIN case=\" << case_id "
        "<< \" iterations=\" << num_iters "
        "<< \" copies=\" << arg_group_count << std::endl;\n",
        "GEMM warmup marker")
    text = replace_once(
        text, "    check_correctness(d_C[0], d_C_ref, M * N);",
        "    if (check) {\n"
        "        check_correctness(d_C[0], d_C_ref, M * N);\n"
        "        std::cout << \"ROTATION_CORRECTNESS_PASS case=\" << case_id "
        "<< \" copies=\" << arg_group_count << \"\\n\";\n"
        "    }",
        "GEMM correctness")
    text = replace_once(
        text, '    std::cout << "Average kernel execution time: " << microseconds << " us\\n";',
        '    std::cout << "Average kernel execution time: " << microseconds << " us\\n";\n'
        '    std::cout << "TK: " << microseconds / 1000.0 << " ms\\n";',
        "GEMM timing")
    main_start = text.index("int main()")
    new_main = r'''int main(int argc, char **argv) {
    if (argc != 9) {
        std::cerr << "usage: gemm CASE M N K ROTATION WARMUP_ROUNDS ITERS CHECK\n";
        return 2;
    }
    std::string case_id = argv[1];
    size_t M = std::stoull(argv[2]);
    size_t N = std::stoull(argv[3]);
    size_t K = std::stoull(argv[4]);
    int rotation = std::stoi(argv[5]);
    int warmup = std::stoi(argv[6]);
    int iterations = std::stoi(argv[7]);
    bool check = std::stoi(argv[8]) != 0;
    if (rotation <= 0 || warmup <= 0 || iterations <= 0) return 2;
    run_benchmark<matmul_template<2,4,8>>(
        M, N, K, rotation, warmup, iterations, check, case_id);
    return 0;
}
'''
    return text[:main_start] + new_main


def build(source, output, tk_root, dry_run):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output = output / ("_C" + suffix)
    print("P1_EXTENSION source={} output={}".format(source, output), flush=True)
    if dry_run or (output.exists() and
                   output.stat().st_mtime >= source.stat().st_mtime):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = "{}:{}".format(pathlib.Path(sys.executable).parent,
                                  env.get("PATH", ""))
    subprocess.run([
        "make", "-f", str(tk_root / "kernels/common.mk"),
        "ARCH=SM90", "CONFIG=pytorch", "CMD=true",
        "SRC={}".format(source), "OUT={}".format(output),
    ], cwd=tk_root / "kernels", env=env, check=True)
    return output


def build_gemm(source, output_root, tk_root, dry_run):
    output = output_root / "gemm" / "bf16_h100_rotation.out"
    print("P1_EXECUTABLE source={} output={}".format(source, output), flush=True)
    if dry_run or (output.exists() and
                   output.stat().st_mtime >= source.stat().st_mtime):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "make", "-f", str(tk_root / "kernels/common.mk"),
        "ARCH=SM90", "CONFIG=standalone", "CMD=true",
        "SRC={}".format(source), "OUT={}".format(output),
    ], cwd=tk_root / "kernels", check=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    tk_root = pathlib.Path(os.environ.get(
        "TK_ROOT", pathlib.Path.home() / "Repos/ThunderKittens")).resolve()
    output_root = args.output_root.resolve()

    mha_original = tk_root / "kernels/attention/mha_h100/mha_h100.cu"
    mha_copy = output_root / "mha/mha_h100_rotation.cu"
    if args.dry_run:
        make_mha_copy(mha_original.read_text(encoding="utf-8"))
    else:
        mha_copy.parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(
            mha_copy,
            make_mha_copy(mha_original.read_text(encoding="utf-8")))
        shutil.copy2(mha_original.parent / "harness.impl",
                     mha_copy.parent / "harness.impl")
        shutil.copy2(mha_original.parent / "benchmark.py",
                     mha_copy.parent / "official_benchmark.py")
    build(mha_copy, output_root / "mha", tk_root, args.dry_run)

    a2a_source = tk_root / "kernels/parallel/all_to_all/all_to_all.cu"
    build(a2a_source, output_root / "a2a", tk_root, args.dry_run)

    gemm_original = tk_root / "kernels/gemm/bf16_h100/bf16_h100_gemm.cu"
    gemm_copy = output_root / "gemm/bf16_h100_rotation.cu"
    if args.dry_run:
        make_gemm_copy(gemm_original.read_text(encoding="utf-8"))
    else:
        gemm_copy.parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(
            gemm_copy,
            make_gemm_copy(gemm_original.read_text(encoding="utf-8")))
        shutil.copy2(gemm_original.parent.parent / "common.cuh",
                     output_root / "common.cuh")
    build_gemm(gemm_copy, output_root, tk_root, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
