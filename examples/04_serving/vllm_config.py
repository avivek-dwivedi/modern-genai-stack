"""
examples/04_serving/vllm_config.py

vLLM serving configuration with prefix caching, speculative decoding,
and PagedAttention. Maps to docs/02-attention-serving/README.md.

NOTE: This is a config example. The actual server is started via CLI:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --max-model-len 32768

This file shows the equivalent Python-side config.
"""

from vllm import LLM, SamplingParams


# ---------- Configuration ----------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.90
DTYPE = "bfloat16"


def build_llm() -> LLM:
    """Build a vLLM engine with production-grade serving settings."""
    return LLM(
        model=MODEL_NAME,
        # Serving optimizations
        enable_prefix_caching=True,      # reuse KV cache for shared prefixes
        enable_chunked_prefill=True,     # split long prompts for better scheduling
        # GPU
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        # KV cache
        block_size=16,                   # PagedAttention block size
        # Quantization (optional — for memory-constrained GPUs)
        # quantization="bitsandbytes",   # or "awq", "gptq", "fp8"
        # Speculative decoding (optional — for low-latency decoding)
        # speculative_model="Qwen/Qwen2.5-0.5B-Instruct",
        # num_speculative_tokens=5,
    )


def generate(llm: LLM, prompts: list[str]) -> list[str]:
    """Generate completions with sampling controls."""
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
        stop=["</answer>", "\n\nUser:"],
    )
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


# ---------- Run ----------

if __name__ == "__main__":
    llm = build_llm()
    prompts = [
        "Explain PagedAttention in one sentence.",
        "What is the difference between GQA and MLA?",
    ]
    for prompt, answer in zip(prompts, generate(llm, prompts)):
        print(f"\nPrompt: {prompt}\nAnswer: {answer}\n")