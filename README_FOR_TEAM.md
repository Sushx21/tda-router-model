# Tata Steel TDA Agent Router (Fine-Tuned Qwen3-4B LoRA)

This package contains the fine-tuned LoRA adapter (`tda_lora_model/`) and production inference script (`router.py`) for routing Tata Steel internal data queries to the correct domain agent with reasoning.

## Supported Agents
| Agent Name | Domain Scope |
| :--- | :--- |
| **`EmployeeInformationAgent`** | Employee master data: names, IDs, grades, departments, gender, tenure, retirement, reporting manager, BUHR/Locational HR contacts. |
| **`ConnectedWorkforceAgent`** | Gate/biometric entry & exit, safety permits (E-Permits), beacon tracking, overstay violations, vendor/contractor and skill/trade details. |
| **`SCSteelInventoryAgent`** | Steel stock and inventory levels, value, age, plan vs actual. |
| **`RMProductionAgent`** | Iron ore and coal production volumes vs plan (ABP/OSP). |

---

## Folder Contents
```text
├── tda_lora_model/              # Fine-tuned LoRA adapter weights & tokenizer (Rank 8, Alpha 16)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
├── router.py                    # Ready-to-use Python module & interactive CLI
├── requirements.txt             # Python dependencies
└── README_FOR_TEAM.md           # This guide
```

---

## Quick Start (Local GPU Server or Google Colab)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# Optional (recommended on Linux/Colab for 2x faster inference):
pip install unsloth
```

### 2. Run the Interactive Router & Sanity Check
```bash
python router.py
```
This will automatically download the base 4-bit weights (`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`) on first run, attach your local `./tda_lora_model` adapter, run 4 sanity-check queries across all 4 agents, and open an interactive prompt.

### 3. Use Inside Any Python Application / API
```python
from router import TDARouter

router = TDARouter()  # Loads base model + ./tda_lora_model once at startup

response = router.route("What is the total quantity and value of 'Flats' classification inventory today?")
print(response)
# Output:
# {
#   "agent": "SCSteelInventoryAgent",
#   "reasoning": "Rationale: Identifies the total stock value of flats today."
# }
```

---

## Important Implementation Notes (Why `router.py` Is Configured This Way)
If integrating the adapter into a custom pipeline without `router.py`, keep these 3 rules in mind:
1. **Use the Exact 643-Char System Prompt:** The LoRA adapter was trained with `train_on_responses_only` conditioned on the exact system prompt defined in `SYSTEM_PROMPT` inside `router.py`. Omitting the agent definitions from the system prompt will degrade accuracy.
2. **Include `<think>\n\n</think>\n\n` in the Assistant Generation Prefix:** Qwen3-4B's chat template includes an empty `<think>\n\n</think>\n\n` block before the JSON response during training. `tda_lora_model/chat_template.jinja` and `router.py` already include this automatically.
3. **Pre-seed `{"agent": "` and Suppress Special Tags:** `router.py` pre-seeds `{"agent": "` and blocks `<tool_call>` / `<|im_start|>` tokens via `suppress_tokens` so the model is guaranteed to return 100% valid JSON on every call.
