#!/usr/bin/env python3
"""
Concurrency x Context Matrix Benchmark for B70 MoE XPU Graph FULL server.

Usage:
  python3 matrix_bench.py --concurrency 1 --target-ctx 4000 --max-tokens 256 --rounds 3
  python3 matrix_bench.py --concurrency 2 --target-ctx 16000
  python3 matrix_bench.py --concurrency 4 --target-ctx 32000

Writes results to stdout in a parseable format.
"""

import asyncio
import aiohttp
import time
import json
import argparse
import sys


def make_prompt(target_tokens):
    """Generate a prompt that fills approximately target_tokens of context.
    ~1.3 tokens per word, so ~0.77 words per token.
    """
    # Base sentence is 10 words ~ 13 tokens
    base = "The distributed system processes incoming data through a multi-stage computational pipeline. "
    base_tokens_est = 13

    repeats_needed = max(1, int(target_tokens / base_tokens_est))
    padding = base * repeats_needed

    prompt = f"""Analyze the following technical documentation carefully:

{padding}

Based on the complete document above, write exactly 200 words summarizing the key architectural concepts. Be specific and concise."""

    return prompt


async def send_request(session, prompt, max_tokens, request_id):
    """Send one streaming request and measure timing."""
    start = time.time()
    first_token_time = None
    output_tokens = 0
    prompt_tokens = 0
    full_content = ""

    payload = {
        "model": "Qwen3.6-35B-A3B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        async with session.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "error": f"HTTP {resp.status}: {body[:200]}",
                    "request_id": request_id,
                }

            async for line in resp.content:
                text = line.decode().strip()
                if not text.startswith("data: "):
                    continue
                if text == "data: [DONE]":
                    break
                try:
                    chunk = json.loads(text[6:])
                    # Check for usage in the final chunk
                    if "usage" in chunk and chunk["usage"]:
                        prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                        output_tokens = chunk["usage"].get("completion_tokens", output_tokens)

                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        output_tokens += len(content.split())  # rough token count
                        full_content += content
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

    except asyncio.TimeoutError:
        return {"error": "timeout (600s)", "request_id": request_id}
    except Exception as e:
        return {"error": str(e), "request_id": request_id}

    end = time.time()
    total = end - start
    ttft = (first_token_time - start) if first_token_time else total
    decode_time = (end - first_token_time) if first_token_time else 0.001

    # Use actual output token count from usage if available, else estimate
    # rough estimate: ~1.3 tokens per word
    est_output_tokens = max(1, int(len(full_content.split()) * 1.3))
    if prompt_tokens == 0:
        # Estimate prompt tokens
        prompt_tokens = int(len(prompt.split()) * 1.3)

    tps = est_output_tokens / decode_time if decode_time > 0 else 0

    return {
        "request_id": request_id,
        "ttft": ttft,
        "tps": tps,
        "output_tokens": est_output_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time": total,
        "decode_time": decode_time,
    }


async def bench_cell(concurrency, target_ctx, max_tokens, rounds):
    """Benchmark one (concurrency, context) cell."""
    prompt = make_prompt(target_ctx)

    print(f"\n{'='*60}", flush=True)
    print(f"CELL: c={concurrency} ctx={target_ctx} max_tokens={max_tokens}", flush=True)
    print(f"Prompt length: ~{len(prompt.split())} words (~{int(len(prompt.split())*1.3)} est tokens)", flush=True)
    print(f"{'='*60}", flush=True)

    all_rounds = []

    for round_num in range(rounds):
        print(f"\n--- Round {round_num+1}/{rounds} ---", flush=True)

        start = time.time()
        connector = aiohttp.TCPConnector(limit=concurrency + 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i in range(concurrency):
                tasks.append(send_request(session, prompt, max_tokens, i))
            responses = await asyncio.gather(*tasks)

        wall_time = time.time() - start

        successes = [r for r in responses if "error" not in r]
        errors = [r for r in responses if "error" in r]

        if successes:
            total_output = sum(r["output_tokens"] for r in successes)
            agg_tps = total_output / wall_time
            avg_per_req_tps = sum(r["tps"] for r in successes) / len(successes)
            avg_ttft = sum(r["ttft"] for r in successes) / len(successes)
            max_ttft = max(r["ttft"] for r in successes)
            avg_prompt_tokens = sum(r["prompt_tokens"] for r in successes) / len(successes)
        else:
            total_output = 0
            agg_tps = 0
            avg_per_req_tps = 0
            avg_ttft = 0
            max_ttft = 0
            avg_prompt_tokens = 0

        round_result = {
            "round": round_num,
            "successes": len(successes),
            "errors": len(errors),
            "wall_time": wall_time,
            "aggregate_tps": agg_tps,
            "per_request_tps": avg_per_req_tps,
            "avg_ttft": avg_ttft,
            "max_ttft": max_ttft,
            "total_output_tokens": total_output,
            "avg_prompt_tokens": avg_prompt_tokens,
            "error_details": [r.get("error", "") for r in errors],
        }
        all_rounds.append(round_result)

        print(f"  Successes: {len(successes)}/{concurrency}", flush=True)
        if errors:
            for e in errors:
                print(f"  ERROR: {e.get('error', 'unknown')}", flush=True)
        print(f"  Wall time: {wall_time:.2f}s", flush=True)
        print(f"  Aggregate tok/s: {agg_tps:.1f}", flush=True)
        print(f"  Per-request tok/s: {avg_per_req_tps:.1f}", flush=True)
        print(f"  Avg TTFT: {avg_ttft:.3f}s", flush=True)
        print(f"  Max TTFT: {max_ttft:.3f}s", flush=True)
        print(f"  Avg prompt tokens: {avg_prompt_tokens:.0f}", flush=True)

    # Summary
    valid_rounds = [r for r in all_rounds if r["successes"] > 0]
    if valid_rounds:
        avg_agg_tps = sum(r["aggregate_tps"] for r in valid_rounds) / len(valid_rounds)
        avg_per_req = sum(r["per_request_tps"] for r in valid_rounds) / len(valid_rounds)
        avg_ttft_all = sum(r["avg_ttft"] for r in valid_rounds) / len(valid_rounds)
        max_ttft_all = max(r["max_ttft"] for r in valid_rounds)
        avg_prompt = sum(r["avg_prompt_tokens"] for r in valid_rounds) / len(valid_rounds)
    else:
        avg_agg_tps = 0
        avg_per_req = 0
        avg_ttft_all = 0
        max_ttft_all = 0
        avg_prompt = 0

    total_errors = sum(r["errors"] for r in all_rounds)

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY: c={concurrency} ctx={target_ctx}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"concurrency:          {concurrency}", flush=True)
    print(f"target_context:       {target_ctx}", flush=True)
    print(f"actual_prompt_tokens: {avg_prompt:.0f}", flush=True)
    print(f"max_tokens:           {max_tokens}", flush=True)
    print(f"rounds:               {rounds}", flush=True)
    print(f"total_errors:         {total_errors}", flush=True)
    print(f"aggregate_tok_s:      {avg_agg_tps:.1f}", flush=True)
    print(f"per_request_tok_s:    {avg_per_req:.1f}", flush=True)
    print(f"avg_ttft:             {avg_ttft_all:.3f}", flush=True)
    print(f"max_ttft:             {max_ttft_all:.3f}", flush=True)
    print(f"{'='*60}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Concurrency x Context Matrix Benchmark")
    parser.add_argument("--concurrency", "-c", type=int, required=True)
    parser.add_argument("--target-ctx", type=int, required=True,
                       help="Target context length in tokens")
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Max output tokens per request (default: 256)")
    parser.add_argument("--rounds", type=int, default=3,
                       help="Number of measurement rounds (default: 3)")
    args = parser.parse_args()

    asyncio.run(bench_cell(args.concurrency, args.target_ctx, args.max_tokens, args.rounds))


if __name__ == "__main__":
    main()
