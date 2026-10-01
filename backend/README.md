# 🧠 JamesBot Local AI Matrix Dashboard (v4.0)

A lightweight, full-stack conversational dashboard that converts a Python rule script into a **real artificial neural network running entirely from scratch using NumPy**. 

This repository requires **no API keys, no internet connection, and zero external hosting services**. It executes pure matrix multiplication, feedforward propagation, and backpropagation optimization algorithms straight out of local host memory.

## 📁 Repository Structure

```text
ai_trainer/
│
├── app.py                 # FastAPI Backend (Secure Matrix Processor)
├── james.py               # Neural AI Engine Engine (Pure NumPy Operations)
├── .gitignore             # Git File Exclusion Configuration
├── README.md              # Repository Overview & Operational Manual
├── static/
│   ├── style.css          # High-Contrast High-Glow Terminal Layout Sheet
│   └── script.js          # Asynchronous Pipeline & Telemetry Controller
└── templates/
    └── index.html         # Frontfacing Multi-Node Visual Interface
```

## 🚀 Quickstart Guide

### 1. Clone the Workspace
```bash
git clone <your-github-repository-url>
cd ai_trainer
```

### 2. Install Hardware Layer Dependencies
Ensure you have the required mathematical array manipulation and micro-framework modules compiled:
```bash
pip install numpy fastapi uvicorn
```

### 3. Initialize the Hidden Neural Engine
Fire up your background environment server:
```bash
uvicorn app:app --reload
```

Open your browser to `http://127.0.0.1:8000` to interact with James's synaptic weights live.

## 🛠️ Matrix Deep Learning Pipeline

* **Input Layer:** Translates raw English text character strings into vectorized binary Bag-of-Words matrices.
* **Hidden Network Layer:** Forwards weights across an 8-neuron hidden processing grid powered by custom `Sigmoid` squashing math functions.
* **Output Classification Layer:** Generates contextual confidence probability percentages across 4 specialized conversational domains (*Greetings, Tech, Gaming, School*).
* **Live Optimizer:** Executes local gradient descent backpropagation loops every turn you chat to realign weight biases in real-time.
