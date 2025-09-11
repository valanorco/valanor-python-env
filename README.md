# Fast Python Environment Setup with uv

This repository documents **how we bootstrap Python environments at [Valanor](https://valanor.co)**.  
It uses [uv](https://astral.sh/uv) for ultra-fast installs and a clear tiered setup.

---

## ✨ Features

- 🚀 Fast installs with `uv`
- 🧩 Three tiers:
  - **minimal** → lean base stack
  - **standard** → DE/DS workhorse (default)
  - **full** → complete DE + ML + DS stack
- 🎛️ Optional GPU installs (Torch, TensorFlow, JAX) with CUDA detection
- 📓 Auto-generated `ENVIRONMENT.md` report with versions
- 🔌 Auto-registered Jupyter kernel for VS Code & JupyterLab

---

## 🚀 Quick Start

```bash
chmod +x init_env.sh
./init_env.sh
```

This will:

- Create a `.venv` (via uv)  
- Install the **standard** stack by default  
- Generate:
  - `requirements.txt` (only for the selected tier)  
  - `ENVIRONMENT.md` (documenting the environment state)  

---

## 🔧 Examples

**Minimal stack:**
```bash
./init_env.sh -t minimal
```

**Full stack with GPU frameworks:**
```bash
./init_env.sh -t full --gpu all
```

**Force specific CUDA wheels:**
```bash
./init_env.sh --gpu torch --cuda cu126
```

---

## ▶️ Activating

```bash
source .venv/bin/activate
```

Deactivate with:
```bash
deactivate
```

---

## 📓 Reports

Every run generates an `ENVIRONMENT.md` like:

```markdown
# Environment Report

**Generated:** 2025-09-10 11:45:03

- Tier: standard
- GPU Frameworks: torch
- CUDA: cu126

## Installed Packages
- numpy==2.1.0
- pandas==2.2.2
- ...
```

---

## ⚙️ Using the Makefile

This repository also includes a **Makefile** to simplify service management.  
It is a **starting point** — you can customize it based on your needs (change ports, add/remove services, or integrate Docker).

### Available commands

```bash
make setup        # run init_env.sh and create environment
make api          # start FastAPI demo app
make jupyter      # launch JupyterLab
make mlflow       # run MLflow UI

make start-all    # start API, Jupyter, MLflow in background
make logs         # tail logs from all services
make stop-all     # stop background services
```

Logs are stored in `./logs` and process IDs in `./run`.

---


## 📜 License

MIT © 2025 **VALANOR DOO**
