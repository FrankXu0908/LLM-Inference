import asyncio
import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import httpx
from model_config import load_model_name


@dataclass
class RequestResult:
    request_id: int
    success: bool
    error: str
    concurrency: int
    prompt_chars: int
    output_chars: int
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    ttft: float
    latency: float
    output_tps: float
    total_tps: float
    start_ts: float
    end_ts: float
    model: str


class TokenCounter:
    def __init__(self, tokenizer_path: Optional[str] = None):
        self.tokenizer = None
        if tokenizer_path:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                use_fast=True,
            )

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # fallback: very rough estimate
        return max(1, int(len(text) / 4))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    semaphore: asyncio.Semaphore,
    request_id: int,
    concurrency: int,
    token_counter: TokenCounter,
    stream: bool = True,
    timeout_s: float = 300.0,
) -> RequestResult:
    async with semaphore:
        start_ts = time.perf_counter()
        first_token_ts = None
        output_text_parts = []
        error_msg = ""
        success = False

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        try:
            if stream:
                async with client.stream("POST", url, json=payload, timeout=timeout_s) as resp:
                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue

                        data_str = line[len("data: "):].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        piece = delta.get("content", "")
                        if piece:
                            if first_token_ts is None:
                                first_token_ts = time.perf_counter()
                            output_text_parts.append(piece)

                success = True
            else:
                resp = await client.post(url, json=payload, timeout=timeout_s)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                output_text_parts.append(text)
                first_token_ts = start_ts
                success = True

        except Exception as e:
            error_msg = repr(e)

        end_ts = time.perf_counter()
        output_text = "".join(output_text_parts)

        prompt_tokens = token_counter.count(prompt)
        output_tokens = token_counter.count(output_text)
        total_tokens = prompt_tokens + output_tokens

        ttft = 0.0
        if first_token_ts is not None:
            ttft = first_token_ts - start_ts

        latency = end_ts - start_ts

        output_tps = 0.0
        if latency > 0 and output_tokens > 0:
            output_tps = output_tokens / latency

        total_tps = 0.0
        if latency > 0 and total_tokens > 0:
            total_tps = total_tokens / latency

        return RequestResult(
            request_id=request_id,
            success=success,
            error=error_msg,
            concurrency=concurrency,
            prompt_chars=len(prompt),
            output_chars=len(output_text),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            ttft=ttft,
            latency=latency,
            output_tps=output_tps,
            total_tps=total_tps,
            start_ts=start_ts,
            end_ts=end_ts,
            model=model,
        )


def _repeat_to_target(unit: str, target_chars: int) -> str:
    text = unit
    while len(text) < target_chars:
        text += unit
    return text[:target_chars]


def build_prompt(prompt_type: str, approx_input_tokens: int) -> str:
    target_chars = max(64, approx_input_tokens * 4)

    if prompt_type == "short":
        unit = (
            "You are a concise assistant. "
            "Answer the user question in 3 bullet points with one actionable suggestion. "
        )
        return _repeat_to_target(unit, target_chars)

    if prompt_type == "medium":
        unit = (
            "In vLLM, the scheduler must balance prefill and decode workloads. "
            "Explain how long prompts can affect TTFT for short interactive queries. "
        )
    elif prompt_type == "long":
        unit = (
            "You are analyzing an LLM inference system. "
            "Discuss the interaction among tensor parallelism, KV cache growth, chunked prefill, "
            "and request scheduling under mixed short and long prompts. "
            "Use concrete system-level reasoning instead of generic advice. "
        )
    elif prompt_type == "rag":
        unit = (
            "Context chunk: The retrieval system returns passages from product manuals, "
            "incident reports, and architecture notes. The answer must cite evidence and avoid hallucination. "
        )
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    return _repeat_to_target(unit, target_chars)


def summarize_results(results: List[RequestResult]) -> dict:
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    if not ok:
        return {
            "total_requests": len(results),
            "success_requests": 0,
            "failed_requests": len(fail),
        }

    latencies = [r.latency for r in ok]
    ttfts = [r.ttft for r in ok if r.ttft > 0]
    output_tps = [r.output_tps for r in ok if r.output_tps > 0]
    total_output_tokens = sum(r.output_tokens for r in ok)

    wall_clock_start = min(r.start_ts for r in ok)
    wall_clock_end = max(r.end_ts for r in ok)
    wall_clock = max(1e-9, wall_clock_end - wall_clock_start)
    aggregate_output_tps = total_output_tokens / wall_clock

    return {
        "total_requests": len(results),
        "success_requests": len(ok),
        "failed_requests": len(fail),
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "latency_mean": statistics.mean(latencies),
        "ttft_p50": percentile(ttfts, 0.50) if ttfts else 0.0,
        "ttft_p95": percentile(ttfts, 0.95) if ttfts else 0.0,
        "ttft_mean": statistics.mean(ttfts) if ttfts else 0.0,
        "output_tps_p50": percentile(output_tps, 0.50) if output_tps else 0.0,
        "output_tps_p95": percentile(output_tps, 0.95) if output_tps else 0.0,
        "output_tps_mean": statistics.mean(output_tps) if output_tps else 0.0,
        "aggregate_output_tps": aggregate_output_tps,
        "total_output_tokens": total_output_tokens,
        "wall_clock": wall_clock,
    }


async def run_benchmark(args):
    token_counter = TokenCounter(args.tokenizer)

    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    prompts = [
        build_prompt(args.prompt_type, args.input_tokens)
        for _ in range(args.num_requests)
    ]

    limits = httpx.Limits(
        max_keepalive_connections=args.concurrency,
        max_connections=args.concurrency * 2,
    )

    timeout = httpx.Timeout(
        connect=30.0,
        read=args.timeout,
        write=30.0,
        pool=30.0,
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout) as client:
        tasks = []
        for i, prompt in enumerate(prompts):
            tasks.append(
                asyncio.create_task(
                    one_request(
                        client=client,
                        url=url,
                        model=args.model,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        semaphore=semaphore,
                        request_id=i,
                        concurrency=args.concurrency,
                        token_counter=token_counter,
                        stream=not args.no_stream,
                        timeout_s=args.timeout,
                    )
                )
            )

        results = await asyncio.gather(*tasks)

    summary = summarize_results(results)

    print("\n=== Benchmark Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        payload = {
            "args": vars(args),
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved detailed results to: {args.output_json}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")

    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--prompt-type", type=str, choices=["short", "medium", "long", "rag"], default="short")
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)

    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--parallel-mode",
        type=str,
        choices=["tp2", "dp2", "dp2_ep", "custom"],
        default="custom",
        help="Tag this benchmark as TP=2 / DP=2 / DP=2+EP (or custom) for result organization.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/tables",
        help="Root directory for auto-saved result JSON when --output-json is not set.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model is None:
        args.model = load_model_name(args.model_config)
    if args.output_json is None:
        model_name = Path(args.model).name
        auto_dir = Path(args.results_root) / model_name / args.parallel_mode
        auto_file = (
            f"bench_c{args.concurrency}_in{args.input_tokens}"
            f"_out{args.max_tokens}_n{args.num_requests}.json"
        )
        args.output_json = str(auto_dir / auto_file)
    asyncio.run(run_benchmark(args))
