---
title: MNIST Digit Classifier Gradio Lite
emoji: 🔢
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
---

# 🔢 MNIST Digit Classifier (Gradio-Lite)

A completely serverless handwritten digit recognition web application powered by **Gradio-Lite (`@gradio/lite`)** and **Pyodide (WebAssembly)**. 

Runs **100% in the user's browser client-side** with **zero server costs**, no active backend container, and instant inference!

---

## 🌟 Key Features
- ⚡ **Pure Client-Side WebAssembly**: Executed in-browser using Pyodide (Wasm) without calling any external compute backend.
- ✍️ **Interactive Drawing Canvas**: Draw numbers (0 to 9) on an interactive sketchpad.
- 📁 **Image Upload**: Upload `.png`, `.jpg`, or `.jpeg` images of handwritten digits with automatic background polarity inversion and normalization.
- 🎲 **Preset Benchmark Samples**: Instant testing with real test set samples from the standard MNIST dataset.
- 📊 **Confidence & Class Probability Chart**: Live probability scores across all 10 digits (0–9).
- 🪶 **Optimized Neural Network**: 3-layer MLP (`784 ➔ 128 (ReLU) ➔ 64 (ReLU) ➔ 10 (Softmax)`) achieving **97.1% benchmark accuracy**.

---

## 🚀 How This Space Works on Hugging Face (Static SDK)

This Space is configured as a **Static HTML Space** (`sdk: static`):
1. **`index.html`**: Entry point that loads `@gradio/lite` from CDN and runs the Python runtime directly in the browser via WebAssembly.
2. **`app.py`**: The Gradio application containing UI layout, pre-processing, forward pass inference, and sample loader.
3. **`model_weights_b64.py`**: Embedded, compressed neural network weights (~440 KB) loaded synchronously with zero external network lag.

---

## 💻 Local Testing & Preview

To run and preview Gradio-Lite locally in your browser:

```bash
# Using Python's built-in HTTP server:
python -m http.server 8000
```
Then open `http://localhost:8000` in your web browser.

You can also run standard Gradio directly in Python:
```bash
python app.py
```
