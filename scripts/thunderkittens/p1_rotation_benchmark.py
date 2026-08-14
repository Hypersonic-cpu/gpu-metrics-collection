#!/usr/bin/env python3
"""Independent rotating copies of the P1 MHA/A2A benchmark paths."""

from rotating_p0_benchmark import (_assert_close, _fingerprint, _run_rotating,
                                   _tensor_bytes, run_ag, run_rs)


def run_mha(module, torch, case, rank, copies, seed_base, warmup_rounds,
            iterations, check_correctness):
    shape = case["execution"]
    device = "cuda:{}".format(rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_base + rank)
    buffers = []
    for _ in range(copies):
        q = torch.randn(
            shape["B"], shape["q_heads"], shape["q_len"], shape["D"],
            dtype=torch.bfloat16, device=device, generator=generator)
        k = torch.randn(
            shape["B"], shape["kv_heads"], shape["kv_len"], shape["D"],
            dtype=torch.bfloat16, device=device, generator=generator)
        v = torch.randn(
            shape["B"], shape["kv_heads"], shape["kv_len"], shape["D"],
            dtype=torch.bfloat16, device=device, generator=generator)
        output = torch.empty_like(q)
        l_vec = torch.empty(
            shape["B"], shape["q_heads"], shape["q_len"], 1,
            dtype=torch.float32, device=device)
        buffers.append({
            "q": q, "k": k, "v": v, "output": output, "l_vec": l_vec,
        })

    def call_at(index):
        item = buffers[index]
        module.mha_forward_into(
            item["q"], item["k"], item["v"], item["output"], item["l_vec"],
            shape["causal"])

    if check_correctness:
        for index, item in enumerate(buffers):
            call_at(index)
            torch.cuda.synchronize()
            first = item["output"].clone()
            call_at(index)
            torch.cuda.synchronize()
            torch.testing.assert_close(item["output"], first, rtol=0, atol=0)
            if not torch.isfinite(item["output"]).all():
                raise RuntimeError("MHA output contains non-finite values")
        print("ROTATION_CORRECTNESS_PASS case={} copies={} mode=repeatability"
              .format(case["case_id"], copies), flush=True)

    item = buffers[0]
    bytes_per_copy = sum(_tensor_bytes(item[name]) for name in (
        "q", "k", "v", "output", "l_vec"))
    fingerprints = [
        _fingerprint(buffers[0]["k"]), _fingerprint(buffers[-1]["k"])]
    average_ms = _run_rotating(
        case["case_id"], rank, copies, warmup_rounds, iterations, call_at,
        bytes_per_copy, fingerprints)
    print("TK: {:.3f} ms | MHA q={} kv={} copies={}".format(
        average_ms, shape["q_len"], shape["kv_len"], copies), flush=True)


def run_a2a(benchmark, case, rank, world_size, copies, seed_base,
            warmup_rounds, iterations, check_correctness):
    torch = benchmark.torch
    shape = case["execution"]
    device = "cuda:{}".format(rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_base + rank)
    buffers = []
    input_shape = (1, shape["N"] // world_size, shape["H"], shape["D"])
    output_shape = (1, shape["N"], shape["H"] // world_size, shape["D"])
    for _ in range(copies):
        input_tensor = benchmark.TKParallelTensor(
            input_shape, dtype=torch.bfloat16, local_rank=rank,
            local_world_size=world_size, multicast=False)
        torch.randn(input_shape, out=input_tensor.data_, generator=generator)
        output_tensor = benchmark.TKParallelTensor(
            output_shape, dtype=torch.bfloat16, local_rank=rank,
            local_world_size=world_size, multicast=False)
        output_tensor.data_.zero_()
        barrier = benchmark.TKParallelTensor(
            (1, 1), dtype=torch.int, local_rank=rank,
            local_world_size=world_size, multicast=True)
        barrier.data_.zero_()
        buffers.append({
            "input": input_tensor, "output": output_tensor,
            "barrier": barrier,
        })

    def call_at(index):
        item = buffers[index]
        benchmark.tk_all_to_all(
            item["output"], item["input"], item["barrier"],
            shape["scatter_axis"], shape["gather_axis"])

    if check_correctness:
        reference = torch.empty(output_shape, dtype=torch.bfloat16, device=device)
        for index, item in enumerate(buffers):
            benchmark.nccl_all_to_all_func(
                reference, item["input"].data_, world_size,
                shape["scatter_axis"], shape["gather_axis"])
            call_at(index)
            torch.cuda.synchronize()
            _assert_close(
                item["output"].data_, reference, case["case_id"], rank,
                index, copies)
        torch.distributed.barrier()
        if rank == 0:
            print("ROTATION_CORRECTNESS_PASS case={} copies={}".format(
                case["case_id"], copies), flush=True)

    item = buffers[0]
    bytes_per_copy = sum(_tensor_bytes(tensor) for tensor in (
        item["input"].data_, item["output"].data_, item["barrier"].data_))
    fingerprints = [
        _fingerprint(buffers[0]["input"].data_),
        _fingerprint(buffers[-1]["input"].data_)]
    average_ms = _run_rotating(
        case["case_id"], rank, copies, warmup_rounds, iterations, call_at,
        bytes_per_copy, fingerprints)
    benchmark.clean_print("TK: {:.3f} ms | A2A copies={}".format(
        average_ms, copies))
