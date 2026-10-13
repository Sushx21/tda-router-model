# Tata Steel TDA Agent Router — A100 VM / RDP Edition (`vm-rdp` branch)

This branch is specifically configured for running the fine-tuned Tata Steel TDA Agent Router on a **Remote Desktop (RDP) VM equipped with an NVIDIA A100 GPU**.

## Why This Branch Is Different from `main` (A100 RDP VM Optimizations)
1. **Full Native 16-Bit (`bfloat16`) on A100 Instead of 4-Bit Quantization:**
   - The `main` branch uses Unsloth/bitsandbytes 4-bit quantization (`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`) designed for small Colab GPUs. On Windows Server RDP VMs, `unsloth` and `bitsandbytes` frequently fail due to Linux-specific Triton/CUDA DLL dependencies.
   - An **NVIDIA A100 (40GB / 80GB VRAM)** easily fits the entire 4B model in native **`torch.bfloat16` (~8 GB VRAM)** with zero quantization accuracy loss and native A100 Tensor Core (`TF32` + `bfloat16` + PyTorch `SDPA` FlashAttention) acceleration.
2. **In-VRAM LoRA Weight Fusion (`merge_and_unload()`):**
   - At startup, `router.py` loads `Qwen/Qwen3-4B-Instruct-2507` + `./tda_lora_model` and fuses the LoRA adapter matrices directly into the base weights in A100 VRAM (`merge_and_unload()`), eliminating LoRA adapter overhead on every request (~150–250 ms response time).
3. **Windows CUDA PyTorch Installer (`setup_vm.bat`):**
   - On Windows RDP VMs, running plain `pip install torch` installs the **CPU-only** version of PyTorch from PyPI. Running `setup_vm.bat` installs the official **CUDA 12.4 wheel** so PyTorch utilizes the A100 GPU.
4. **Local Model Cache (`./model_cache`) & FastAPI Server (`api_server.py`):**
   - Stores downloaded base weights in `./model_cache` inside the project folder so restricted RDP user profiles (`C:\Users\<user>`) do not run out of quota.
   - Includes `api_server.py` to expose `POST http://0.0.0.0:8000/route` as an internal microservice.

---

## Quick Start on the A100 RDP VM

### Step 1: Clone the `vm-rdp` Branch
```bash
git clone -b vm-rdp https://github.com/Sushx21/tda-router-model.git
cd tda-router-model
```

### Step 2: Install Dependencies (With A100 CUDA Support)
**On Windows RDP (Command Prompt / PowerShell):**
```cmd
setup_vm.bat
```
*(Or manually on Windows/Linux:)*
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Step 3A: Run Interactive CLI & Sanity Check
```bash
python router.py
```
*(Or double-click `run_router.bat` on the RDP Desktop.)*

### Step 3B: Run as a Production REST API Server (Port 8000)
```bash
python api_server.py
```
Test the API from PowerShell or any client:
```bash
curl -X POST http://localhost:8000/route -H "Content-Type: application/json" -d "{\"query\": \"list employees that may retire in next 2 months\"}"
```
Response:
```json
{
  "agent": "EmployeeInformationAgent",
  "reasoning": "Identifies employees whose retirement dates fall within the next 2 months.",
  "latency_ms": 184.2
}
```
