"""
Tata Steel TDA Agent Router — Production Inference Module
Base Model : unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit (or Qwen/Qwen3-4B-Instruct-2507)
LoRA Model : ./tda_lora_model
"""

import json
import os
import re
from pathlib import Path
import torch

# Resolve adapter folder relative to this script so it works from any working directory
SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = str(SCRIPT_DIR / "tda_lora_model")
BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

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
    def __init__(self, adapter_dir: str = ADAPTER_DIR, load_in_4bit: bool = True):
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(
                f"LoRA adapter folder not found at: {adapter_dir}\n"
                "Make sure 'tda_lora_model' is unzipped in the same folder as router.py."
            )

        # Use Unsloth if available (fastest), otherwise standard Hugging Face PEFT + Transformers
        try:
            from unsloth import FastLanguageModel

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_dir,
                max_seq_length=512,
                load_in_4bit=load_in_4bit,
                dtype=None,
            )
            FastLanguageModel.for_inference(self.model)
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
            quant_config = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base_model, adapter_dir)
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
        Returns a dict: {"agent": "<AgentName>", "reasoning": "<Why>"}
        """
        prefix = '{"agent": "'
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_query.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{prefix}"
        )
        inputs = self.tokenizer(
            [prompt], return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)

        with torch.no_grad():
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

        try:
            return json.loads(full_json_str)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", full_json_str, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {"agent": "UNKNOWN", "reasoning": full_json_str}


if __name__ == "__main__":
    print("Loading Tata Steel TDA Fine-Tuned Router...")
    router = TDARouter()
    print("✅ Model loaded successfully!\n")

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
