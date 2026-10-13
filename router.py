"""
Tata Steel TDA Agent Router — A100 VM / RDP Edition
Optimized for NVIDIA A100 GPU on Windows/Linux RDP VMs:
- Uses native bfloat16 + PyTorch SDPA (Flash/Memory-Efficient Attention) on A100 Tensor Cores
- Fuses LoRA adapter into base weights in VRAM at startup (merge_and_unload) for zero adapter latency
- No bitsandbytes or unsloth DLL dependencies required on Windows Server / RDP
"""

import json
import os
import re
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Resolve paths relative to this script so it works from any RDP directory or scheduled service
SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = str(SCRIPT_DIR / "tda_lora_model")

# Keep Hugging Face model cache inside the repo folder by default on the VM
DEFAULT_CACHE_DIR = str(SCRIPT_DIR / "model_cache")
os.environ.setdefault("HF_HOME", DEFAULT_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE_MODEL = os.environ.get("TDA_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

# Exact 643-character system prompt used during fine-tuning
SYSTEM_PROMPT = (
    "You are a routing assistant for Tata Steel's internal data-query system. "
    "Given a question, decide which agent should handle it and explain why.\n\n"
    "Agents:\n"
    "- EmployeeInformationAgent: employee master data - names, IDs, grades, departments, gender, tenure, retirement, reporting manager, BUHR/Locational HR contacts.\n"
    "- ConnectedWorkforceAgent: gate/biometric entry and exit, safety permits (E-Permits), beacon tracking, overstay violations, vendor/contractor and skill/trade details.\n"
    "- SCSteelInventoryAgent: steel stock and inventory levels, value, age, plan vs actual.\n"
    "- RMProductionAgent: iron ore and coal production volumes vs plan (ABP/OSP)."
)

VALID_AGENTS = {
    "EmployeeInformationAgent",
    "ConnectedWorkforceAgent",
    "SCSteelInventoryAgent",
    "RMProductionAgent",
}


class TDARouter:
    def __init__(
        self,
        adapter_dir: str = ADAPTER_DIR,
        base_model_name: str = BASE_MODEL,
        cache_dir: str = DEFAULT_CACHE_DIR,
        offline: bool = False,
        merge_lora_in_vram: bool = True,
    ):
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(
                f"LoRA adapter folder not found at: {adapter_dir}\n"
                "Make sure 'tda_lora_model' is in the same directory as router.py."
            )

        os.makedirs(cache_dir, exist_ok=True)
        self.has_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.has_cuda else "cpu"

        if self.has_cuda:
            # Enable A100 TensorFloat-32 and cuDNN benchmark optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"[TDARouter] GPU Detected     : {gpu_name} ({vram_gb:.1f} GB VRAM)")
            print(f"[TDARouter] Precision Mode   : {self.dtype} + SDPA Tensor Core Acceleration")
        else:
            self.dtype = torch.bfloat16
            print("[TDARouter] WARNING: CUDA not detected in PyTorch! Running on CPU.")
            print("            On an A100 VM, ensure CUDA PyTorch is installed (run setup_vm.bat).")

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=offline)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=self.dtype,
            device_map="cuda:0" if self.has_cuda else "cpu",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            local_files_only=offline,
        )

        peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
        if merge_lora_in_vram:
            # Fuse LoRA weights (W + alpha/r * B * A) directly into the 16-bit A100 weights for faster inference
            print("[TDARouter] Merging LoRA adapter into base weights in VRAM for zero-overhead inference...")
            self.model = peft_model.merge_and_unload()
        else:
            self.model = peft_model

        self.model.eval()

        # Silence Hugging Face max_length vs max_new_tokens warning
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None

        # Configure stop tokens (<|im_end|> = 151645)
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = list({t for t in [self.tokenizer.eos_token_id, im_end_id] if isinstance(t, int)})

        # Block special control/tool tokens so the model only emits clean JSON text
        self.bad_tokens = []
        for tok_str in ["<|im_start|>", "<tool_call>", "</tool_call>", "<think>", "</think>"]:
            tid = self.tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                self.bad_tokens.append(tid)

    def route(self, user_query: str, max_new_tokens: int = 256) -> dict:
        """
        Routes a natural-language user query to the appropriate Tata Steel TDA agent.
        Returns a dict: {"agent": "<AgentName>", "reasoning": "<Why>", "latency_ms": <float>}
        """
        t0 = time.perf_counter()
        prefix = '{"agent": "'
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_query.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{prefix}"
        )
        inputs = self.tokenizer(
            [prompt], return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.eos_ids,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                suppress_tokens=self.bad_tokens if self.bad_tokens else None,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        raw_completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        full_json_str = prefix + raw_completion
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        try:
            parsed = json.loads(full_json_str)
            parsed["latency_ms"] = latency_ms
            return parsed
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", full_json_str, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    parsed["latency_ms"] = latency_ms
                    return parsed
                except json.JSONDecodeError:
                    pass
            return {"agent": "UNKNOWN", "reasoning": full_json_str, "latency_ms": latency_ms}


if __name__ == "__main__":
    print("Loading Tata Steel TDA Fine-Tuned Router (A100 VM/RDP Edition)...")
    router = TDARouter()
    print("✅ Model loaded and fused on GPU!\n")

    sample_queries = [
        "What is the total quantity and value of 'Flats' classification inventory today?",
        "list employees that may retire in next 2 months",
        "How many contractor workers had overstay violations at the gate yesterday?",
        "What was the iron ore production volume vs ABP plan this month?",
    ]

    print("=== Running Sample Sanity Checks ===")
    for q in sample_queries:
        res = router.route(q)
        print(f"❓ Query : {q}")
        print(f"🤖 Output: {json.dumps(res)}")
        print("-" * 60)

    print("\n=== Interactive Mode (type 'exit' to quit) ===")
    while True:
        try:
            user_input = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit", "q", ""):
            break
        result = router.route(user_input)
        print(json.dumps(result, indent=2))
