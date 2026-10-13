"""
FastAPI Network Server for Tata Steel TDA Agent Router (A100 VM / RDP Edition)
Exposes a port (default: 8000) across the network (0.0.0.0) so colleagues and
enterprise services can query the A100 router from their own machines:
- Web UI       : http://<VM_IP>:8000/
- Swagger Docs : http://<VM_IP>:8000/docs
- REST Endpoint: POST http://<VM_IP>:8000/route
"""

import argparse
import os
import socket
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from router import TDARouter

app = FastAPI(
    title="Tata Steel TDA Agent Router API",
    description="Fine-tuned Qwen3-4B LoRA router running on NVIDIA A100 GPU VM",
    version="1.0.0",
)

# Allow any internal web app or colleague script on the corporate network to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router_engine: TDARouter | None = None
gpu_lock = threading.Lock()


class RouteRequest(BaseModel):
    query: str


class RouteResponse(BaseModel):
    agent: str
    reasoning: str
    latency_ms: float | None = None


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


@app.on_event("startup")
def load_model():
    global router_engine
    router_engine = TDARouter()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": router_engine.device if router_engine else "loading",
    }


@app.post("/route", response_model=RouteResponse)
def route_query(req: RouteRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if router_engine is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    with gpu_lock:
        return router_engine.route(req.query)


@app.get("/", response_class=HTMLResponse)
def web_playground():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Tata Steel TDA Router (A100 GPU)</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 20px; background: #0f172a; color: #f8fafc; }
    h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
    p.sub { color: #94a3b8; margin-top: 0; }
    textarea { width: 100%; height: 90px; padding: 12px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: #f8fafc; font-size: 1rem; box-sizing: border-box; }
    button { margin-top: 12px; padding: 10px 20px; border: none; border-radius: 8px; background: #2563eb; color: white; font-weight: 600; cursor: pointer; font-size: 0.95rem; }
    button:hover { background: #1d4ed8; }
    pre { background: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #334155; overflow-x: auto; color: #38bdf8; }
    .samples button { background: #334155; margin-right: 6px; margin-top: 6px; padding: 6px 12px; font-size: 0.82rem; }
  </style>
</head>
<body>
  <h1>Tata Steel TDA Agent Router (A100 VM)</h1>
  <p class="sub">Fine-tuned Qwen3-4B LoRA Router &bull; <a href="/docs" style="color:#60a5fa">Swagger API Docs (/docs)</a></p>
  <textarea id="q" placeholder="Ask a Tata Steel data question...">What is the total quantity and value of 'Flats' classification inventory today?</textarea>
  <div class="samples">
    <button onclick="setQ('What is the total quantity and value of \\'Flats\\' classification inventory today?')">Flats Inventory</button>
    <button onclick="setQ('list employees that may retire in next 2 months')">Retiring Employees</button>
    <button onclick="setQ('How many contractor workers had overstay violations at the gate yesterday?')">Gate Overstay</button>
    <button onclick="setQ('What was the iron ore production volume vs ABP plan this month?')">Iron Ore vs ABP</button>
  </div>
  <button onclick="send()">Route Question</button>
  <pre id="out">Click "Route Question" to test...</pre>
  <script>
    function setQ(t) { document.getElementById('q').value = t; }
    async function send() {
      const out = document.getElementById('out');
      out.textContent = 'Routing on A100 GPU...';
      try {
        const res = await fetch('/route', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query: document.getElementById('q').value})
        });
        const data = await res.json();
        out.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        out.textContent = 'Error: ' + e;
      }
    }
  </script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TDA Router HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TDA_PORT", "8000")), help="Port to expose (default: 8000)")
    args = parser.parse_args()

    vm_ip = get_local_ip()
    print("=" * 72)
    print(f"Starting Tata Steel TDA Router Server on port {args.port}...")
    print(f"  Local URL   : http://localhost:{args.port}")
    print(f"  Network URL : http://{vm_ip}:{args.port}      (Share with colleagues on LAN/VPN)")
    print(f"  API Docs    : http://{vm_ip}:{args.port}/docs")
    print(f"  POST Route  : http://{vm_ip}:{args.port}/route")
    print("=" * 72)

    uvicorn.run(app, host=args.host, port=args.port)
