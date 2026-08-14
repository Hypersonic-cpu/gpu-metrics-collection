#!/usr/bin/env python3
"""Buffer-rotating executions derived from TK's official AG and MoE runners."""

import hashlib
import os
import time


GIB = 1024 ** 3


def _fingerprint(tensor):
    """Return a stable digest without copying an entire large tensor to host."""
    import torch

    sample = tensor.detach().reshape(-1)[:4096].cpu().view(torch.uint8)
    return hashlib.sha256(sample.numpy().tobytes()).hexdigest()[:16]


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _assert_close(actual, expected, case_id, rank, copy_index, copies):
    import torch

    difference = (actual.float() - expected.float()).abs()
    maximum = difference.max().item()
    mean = difference.mean().item()
    torch.testing.assert_close(actual, expected, rtol=0.1, atol=1.0)
    if rank == 0:
        print("ROTATION_CHECK_OK case={} copy={}/{} max_diff={:.8f} "
              "mean_diff={:.8f}".format(
                  case_id, copy_index + 1, copies, maximum, mean), flush=True)


def _run_rotating(case_id, rank, copies, warmup_rounds, iterations,
                  call_at, bytes_per_copy, fingerprints):
    import torch

    distributed = (torch.distributed.is_available() and
                   torch.distributed.is_initialized())
    if distributed:
        torch.distributed.barrier()
    allocation_bytes = bytes_per_copy * copies
    if rank == 0:
        print("ROTATION_ALLOC_DONE case={} copies={} bytes_per_copy={} "
              "working_set_bytes_per_gpu={} working_set_gib_per_gpu={:.3f} "
              "fingerprint_first={} fingerprint_last={}".format(
                  case_id, copies, bytes_per_copy, allocation_bytes,
                  allocation_bytes / GIB, fingerprints[0], fingerprints[-1]),
              flush=True)

    warmup_started = time.monotonic()
    for _ in range(warmup_rounds):
        for copy_index in range(copies):
            call_at(copy_index)
    torch.cuda.synchronize()
    if distributed:
        torch.distributed.barrier()
    if rank == 0:
        print("ROTATION_WARMUP_DONE case={} rounds={} kernels={} seconds={:.3f}"
              .format(case_id, warmup_rounds, warmup_rounds * copies,
                      time.monotonic() - warmup_started), flush=True)
        print("ROTATION_PROFILE_READY case={}".format(case_id), flush=True)

    start_file = os.environ.get("ROTATION_START_FILE")
    if start_file:
        while not os.path.exists(start_file):
            time.sleep(0.001)
        if distributed:
            torch.distributed.barrier()
    if rank == 0:
        print("ROTATION_MEASURE_BEGIN case={} iterations={} copies={}".format(
            case_id, iterations, copies), flush=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for iteration in range(iterations):
        call_at(iteration % copies)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms / iterations


def run_ag(benchmark, case_id, shape, num_comm_sms, rank, world_size,
           copies, seed_base, warmup_rounds, iterations, check_correctness):
    torch = benchmark.torch
    device = "cuda:{}".format(rank)
    m, k, n = shape["M"], shape["K"], shape["N"]
    local_rows = m // world_size
    first = rank * local_rows
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_base + rank)
    buffers = []

    for _ in range(copies):
        a_local = torch.randn(
            local_rows, k, dtype=torch.bfloat16, device=device,
            generator=generator) / k ** 0.25
        a_tk = benchmark.TKParallelTensor(
            (m, k), dtype=torch.bfloat16, local_rank=rank,
            local_world_size=world_size, multicast=True)
        a_tk.data_[first:first + local_rows].copy_(a_local)
        b = torch.randn(
            k, n, dtype=torch.bfloat16, device=device,
            generator=generator) / k ** 0.25
        c = torch.zeros(m, n, dtype=torch.bfloat16, device=device)
        barrier = benchmark.TKParallelTensor(
            (2, 1024, 1024), dtype=torch.int, local_rank=rank,
            local_world_size=world_size, multicast=True)
        barrier.data_.zero_()
        buffers.append({"a": a_tk, "b": b, "c": c, "barrier": barrier})

    def call_at(index):
        item = buffers[index]
        benchmark.all_gather_matmul(
            item["a"], item["b"], item["c"], item["barrier"], num_comm_sms)

    if check_correctness:
        a_reference = torch.empty(m, k, dtype=torch.bfloat16, device=device)
        c_reference = torch.empty(m, n, dtype=torch.bfloat16, device=device)
        for index, item in enumerate(buffers):
            torch.distributed.all_gather_into_tensor(
                a_reference, item["a"].data_[first:first + local_rows])
            torch.matmul(a_reference, item["b"], out=c_reference)
            call_at(index)
            torch.cuda.synchronize()
            _assert_close(
                item["c"], c_reference, case_id, rank, index, copies)
        torch.distributed.barrier()
        if rank == 0:
            print("ROTATION_CORRECTNESS_PASS case={} copies={}".format(
                case_id, copies), flush=True)

    first_item = buffers[0]
    bytes_per_copy = sum(_tensor_bytes(tensor) for tensor in (
        first_item["a"].data_, first_item["b"], first_item["c"],
        first_item["barrier"].data_))
    fingerprints = [
        _fingerprint(buffers[0]["b"]), _fingerprint(buffers[-1]["b"])]
    tk_avg_ms = _run_rotating(
        case_id, rank, copies, warmup_rounds, iterations, call_at,
        bytes_per_copy, fingerprints)
    total_tflops = 2.0 * m * n * k * 1e-12
    tk_tflops = total_tflops / (tk_avg_ms * 1e-3)
    benchmark.clean_print(
        "<BF16 TP Matmul rotation | world_size={} | {}x{}x{} | "
        "num_comm_sms={} | copies={} | seed_base={}>".format(
            world_size, m, k, n, num_comm_sms, copies, seed_base),
        print_once=True)
    benchmark.clean_print("TK: {:.3f} ms | {:.2f} TFLOp/s".format(
        tk_avg_ms, tk_tflops))


def run_rs(benchmark, case_id, shape, rank, world_size, copies, seed_base,
           warmup_rounds, iterations, check_correctness):
    torch = benchmark.torch
    device = "cuda:{}".format(rank)
    m, k, n = shape["M"], shape["K"], shape["N"]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_base + rank)
    buffers = []

    for _ in range(copies):
        a = torch.randn(
            m, k, dtype=torch.bfloat16, device=device,
            generator=generator) / k ** 0.25
        b = torch.randn(
            k, n, dtype=torch.bfloat16, device=device,
            generator=generator) / k ** 0.25
        c = benchmark.TKParallelTensor(
            (m // world_size, n), dtype=torch.bfloat16, local_rank=rank,
            local_world_size=world_size, multicast=False)
        c.data_.zero_()
        barrier = benchmark.TKParallelTensor(
            (1, 1), dtype=torch.int, local_rank=rank,
            local_world_size=world_size, multicast=True)
        barrier.data_.zero_()
        buffers.append({"a": a, "b": b, "c": c, "barrier": barrier})

    def call_at(index):
        item = buffers[index]
        benchmark.matmul_reduce_scatter(
            item["a"], item["b"], item["c"], item["barrier"])

    if check_correctness:
        intermediate = torch.empty(m, n, dtype=torch.bfloat16, device=device)
        reference = torch.empty(
            m // world_size, n, dtype=torch.bfloat16, device=device)
        for index, item in enumerate(buffers):
            torch.matmul(item["a"], item["b"], out=intermediate)
            torch.distributed.reduce_scatter_tensor(
                reference, intermediate, op=torch.distributed.ReduceOp.SUM)
            call_at(index)
            torch.cuda.synchronize()
            _assert_close(
                item["c"].data_, reference, case_id, rank, index, copies)
        torch.distributed.barrier()
        if rank == 0:
            print("ROTATION_CORRECTNESS_PASS case={} copies={}".format(
                case_id, copies), flush=True)

    first_item = buffers[0]
    bytes_per_copy = sum(_tensor_bytes(tensor) for tensor in (
        first_item["a"], first_item["b"], first_item["c"].data_,
        first_item["barrier"].data_))
    fingerprints = [
        _fingerprint(buffers[0]["b"]), _fingerprint(buffers[-1]["b"])]
    tk_avg_ms = _run_rotating(
        case_id, rank, copies, warmup_rounds, iterations, call_at,
        bytes_per_copy, fingerprints)
    total_tflops = 2.0 * m * n * k * 1e-12
    tk_tflops = total_tflops / (tk_avg_ms * 1e-3)
    benchmark.clean_print(
        "<BF16 TP Matmul Reduce-Scatter rotation | world_size={} | "
        "{}x{}x{} | copies={} | seed_base={}>".format(
            world_size, m, k, n, copies, seed_base), print_once=True)
    benchmark.clean_print("TK: {:.3f} ms | {:.2f} TFLOp/s".format(
        tk_avg_ms, tk_tflops))


def _make_pull_indices(torch, chosen, padded, rank, world_size):
    total_tokens = chosen.shape[0]
    tokens_per_rank = total_tokens // world_size
    experts_per_rank = padded.numel() // world_size
    expert_start = rank * experts_per_rank
    expert_end = expert_start + experts_per_rank
    rows = int(padded[expert_start:expert_end].sum().item())
    offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=chosen.device),
        torch.cumsum(padded[expert_start:expert_end - 1], dim=0,
                     dtype=torch.int32),
    ])
    pull = torch.full((rows, 2), -1, dtype=torch.int32,
                      device=chosen.device)
    chosen_cpu = chosen.cpu()
    for step in range(world_size):
        source_rank = (step + rank) % world_size
        for source_token in range(tokens_per_rank):
            token = source_rank * tokens_per_rank + source_token
            for expert_tensor in chosen_cpu[token]:
                expert = int(expert_tensor)
                if expert_start <= expert < expert_end:
                    local_expert = expert % experts_per_rank
                    offset = offsets[local_expert]
                    pull[offset, 0] = source_rank
                    pull[offset, 1] = source_token
                    offsets[local_expert] += 1
    return pull


def run_moe(benchmark, case_id, shape, num_comm_sms, rank, world_size,
            copies, seed_base, warmup_rounds, iterations, check_correctness):
    torch = benchmark.torch
    device = "cuda:{}".format(rank)
    b, s = shape["B"], shape["S"]
    h, intermediate = shape["H"], shape["I"]
    experts, top_k = shape["experts"], shape["top_k"]
    local_tokens = b * s // world_size
    experts_per_rank = experts // world_size
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_base + rank)

    def make_input_and_weight():
        inputs_hosted = torch.randn(
            local_tokens, h, dtype=torch.bfloat16, device=device,
            generator=generator) / h ** 0.5
        inputs = benchmark.TKParallelTensor(
            (local_tokens, h), dtype=torch.bfloat16, local_rank=rank,
            local_world_size=world_size, multicast=False)
        inputs.data_.copy_(inputs_hosted)
        weights = torch.randn(
            experts_per_rank, h, intermediate, dtype=torch.bfloat16,
            device=device, generator=generator) / h ** 0.5
        return inputs, weights

    # Preserve the official seed and RNG order for copy zero and routing.
    first_inputs, first_weights = make_input_and_weight()
    if rank == 0:
        routing_weights = torch.rand(
            experts, dtype=torch.float32, device=device, generator=generator)
        chosen = torch.multinomial(
            routing_weights.repeat(b * s, 1), top_k, replacement=False,
            generator=generator).to(torch.int32)
        counts = torch.bincount(
            chosen.reshape(-1), minlength=experts).to(torch.int32)
    else:
        chosen = torch.empty(b * s, top_k, dtype=torch.int32, device=device)
        counts = torch.empty(experts, dtype=torch.int32, device=device)
    torch.distributed.broadcast(chosen, 0)
    torch.distributed.broadcast(counts, 0)
    padded = (counts + 127) // 128 * 128
    padded_by_rank = padded.reshape(world_size, experts_per_rank).sum(dim=1)
    padded_local_tokens = int(padded_by_rank[rank].item())
    padded_max_tokens = int(padded_by_rank.max().item())
    pull = _make_pull_indices(torch, chosen, padded, rank, world_size)

    buffers = []
    for copy_index in range(copies):
        if copy_index == 0:
            inputs, weights = first_inputs, first_weights
        else:
            inputs, weights = make_input_and_weight()
        gathered = torch.zeros(
            padded_local_tokens, h, dtype=torch.bfloat16, device=device)
        outputs = torch.zeros(
            padded_local_tokens, intermediate, dtype=torch.bfloat16,
            device=device)
        padded_copy = padded.clone()
        pull_copy = pull.clone()
        barrier = benchmark.TKParallelTensor(
            (2, max(1, padded_max_tokens)), dtype=torch.int, local_rank=rank,
            local_world_size=world_size, multicast=True)
        barrier.data_.zero_()
        buffers.append({
            "inputs": inputs, "gathered": gathered, "weights": weights,
            "outputs": outputs, "padded": padded_copy, "pull": pull_copy,
            "barrier": barrier,
        })

    def call_at(index):
        item = buffers[index]
        benchmark.moe_dispatch_gemm(
            item["inputs"], item["gathered"], item["weights"],
            item["outputs"], item["padded"], item["pull"], item["barrier"],
            num_comm_sms, padded_local_tokens)

    if check_correctness:
        all_inputs = torch.empty(
            world_size, local_tokens, h, dtype=torch.bfloat16, device=device)
        gathered_reference = torch.empty(
            padded_local_tokens, h, dtype=torch.bfloat16, device=device)
        output_reference = torch.empty(
            padded_local_tokens, intermediate, dtype=torch.bfloat16,
            device=device)
        expert_offset = rank * experts_per_rank
        for index, item in enumerate(buffers):
            gathered_reference.zero_()
            output_reference.zero_()
            torch.distributed.all_gather_into_tensor(
                all_inputs, item["inputs"].data_)
            valid = item["pull"][:, 0] >= 0
            gathered_reference[valid] = all_inputs[
                item["pull"][valid, 0].long(),
                item["pull"][valid, 1].long()]
            start = 0
            for local_expert in range(experts_per_rank):
                rows = int(item["padded"][expert_offset + local_expert].item())
                end = start + rows
                if rows:
                    torch.matmul(
                        gathered_reference[start:end], item["weights"][local_expert],
                        out=output_reference[start:end])
                start = end
            call_at(index)
            torch.cuda.synchronize()
            _assert_close(
                item["outputs"], output_reference, case_id, rank, index,
                copies)
        torch.distributed.barrier()
        if rank == 0:
            print("ROTATION_CORRECTNESS_PASS case={} copies={}".format(
                case_id, copies), flush=True)

    first_item = buffers[0]
    bytes_per_copy = sum(_tensor_bytes(tensor) for tensor in (
        first_item["inputs"].data_, first_item["gathered"],
        first_item["weights"], first_item["outputs"], first_item["padded"],
        first_item["pull"], first_item["barrier"].data_))
    fingerprints = [
        _fingerprint(buffers[0]["weights"]),
        _fingerprint(buffers[-1]["weights"])]
    tk_avg_ms = _run_rotating(
        case_id, rank, copies, warmup_rounds, iterations, call_at,
        bytes_per_copy, fingerprints)
    total_flops = 2.0 * (b * s * top_k) * h * intermediate / world_size
    total_tflops = total_flops * 1e-12
    tk_tflops = total_tflops / (tk_avg_ms * 1e-3)
    benchmark.clean_print(
        "<MoE Dispatch GEMM rotation | world_size={} | num_experts={} | "
        "top_k={} | {}x{}x{}x{} | num_comm_sms={} | copies={} | "
        "seed_base={}>".format(
            world_size, experts, top_k, b, s, h, intermediate,
            num_comm_sms, copies, seed_base), print_once=True)
    benchmark.clean_print("TK: {:.3f} ms | {:.2f} TFLOp/s".format(
        tk_avg_ms, tk_tflops))
