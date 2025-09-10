# Fast Python Environment Setup with uv

This repository documents **how we bootstrap Python environments at Valanor**.  
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

## 🏷️ Valanor Standard

_This is the way we do it at Valanor._

---

## 📜 License

MIT © 2025 **VALANOR DOO**
