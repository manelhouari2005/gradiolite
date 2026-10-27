import base64
import json
import zlib
import numpy as np
from PIL import Image, ImageOps
import gradio as gr
from model_weights_b64 import COMPRESSED_DATA

# Decompress and load model weights directly in memory
raw_json = zlib.decompress(base64.b64decode(COMPRESSED_DATA)).decode('utf-8')
model_data = json.loads(raw_json)

W1 = np.array(model_data['w1'], dtype=np.float32)
B1 = np.array(model_data['b1'], dtype=np.float32)
W2 = np.array(model_data['w2'], dtype=np.float32)
B2 = np.array(model_data['b2'], dtype=np.float32)
W3 = np.array(model_data['w3'], dtype=np.float32)
B3 = np.array(model_data['b3'], dtype=np.float32)

ACCURACY = model_data.get('acc', 0.9707)
SAMPLES = model_data.get('samples', {})

def forward_pass(x_vector):
    # MLP Architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
    h1 = np.maximum(0, np.dot(x_vector, W1) + B1)
    h2 = np.maximum(0, np.dot(h1, W2) + B2)
    logits = np.dot(h2, W3) + B3
    # Stable softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs[0]

def extract_image_array(input_data):
    if input_data is None:
        return None
        
    # Handling dictionary format from gr.Sketchpad or gr.ImageEditor
    if isinstance(input_data, dict):
        if "composite" in input_data and input_data["composite"] is not None:
            img = input_data["composite"]
        elif "layers" in input_data and len(input_data["layers"]) > 0:
            img = input_data["layers"][0]
        elif "background" in input_data and input_data["background"] is not None:
            img = input_data["background"]
        else:
            return None
    else:
        img = input_data

    # Convert to PIL Image
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if np.max(img) <= 1.0 else img.astype(np.uint8)
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img, mode='L')
        elif img.shape[2] == 4:
            # RGBA image: check alpha channel or convert to grayscale
            pil_img = Image.fromarray(img, mode='RGBA')
            # Extract grayscale with transparency awareness
            r, g, b, a = pil_img.split()
            # If drawn on transparent canvas, use alpha as luminance
            if np.max(np.array(a)) > 0:
                pil_img = a
            else:
                pil_img = pil_img.convert('L')
        else:
            pil_img = Image.fromarray(img).convert('L')
    elif isinstance(img, Image.Image):
        pil_img = img.convert('L')
    else:
        return None

    # Invert if mostly white background (e.g. user uploaded black drawing on white)
    arr_check = np.array(pil_img)
    if np.mean(arr_check) > 127:
        pil_img = ImageOps.invert(pil_img)

    # Resize to 28x28
    pil_28 = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
    arr_28 = np.array(pil_28, dtype=np.float32) / 255.0

    # Ensure there is actual content
    if np.max(arr_28) < 0.05:
        return None

    return arr_28

def predict_digit(input_data):
    arr_28 = extract_image_array(input_data)
    if arr_28 is None:
        return (
            "⚠️ Canvas is empty. Draw or upload a digit first!",
            {str(i): 0.0 for i in range(10)},
            None
        )

    x_vec = arr_28.reshape(1, 784)
    probs = forward_pass(x_vec)
    pred_digit = int(np.argmax(probs))
    confidence = probs[pred_digit] * 100

    confidences = {str(i): float(probs[i]) for i in range(10)}
    status_msg = f"🎯 **Predicted Digit: {pred_digit}** (Confidence: {confidence:.1f}%)"
    
    # Return formatted 28x28 preview
    preview_img = Image.fromarray((arr_28 * 255).astype(np.uint8), mode='L')
    return status_msg, confidences, preview_img

def load_sample_digit(digit_str, variant_idx):
    variant_idx = int(variant_idx)
    if digit_str in SAMPLES and len(SAMPLES[digit_str]) > variant_idx:
        vec = SAMPLES[digit_str][variant_idx]
        mat = np.array(vec, dtype=np.float32).reshape(28, 28)
        pil_img = Image.fromarray((mat * 255).astype(np.uint8), mode='L')
        return pil_img
    return None

custom_css = """
body {
    background-color: #0b0f19;
    color: #f1f5f9;
}
.gradio-container {
    max-width: 1050px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.header-box {
    text-align: center;
    padding: 1.8rem 1rem 1.2rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #60A5FA, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-box p {
    font-size: 1.05rem;
    color: #94A3B8;
    margin: 0;
}
.badge-row {
    margin-top: 0.8rem;
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #A5B4FC;
    padding: 0.25rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}
"""

with gr.Blocks(title="MNIST Digit Classifier | Gradio-Lite") as demo:
    gr.HTML(f"""
    <style>
    {custom_css}
    </style>
    <div class="header-box">
        <h1>🔢 MNIST Digit Classifier</h1>
        <p>Serverless neural network digit recognition running 100% in your browser via WebAssembly (Gradio-Lite / Pyodide)</p>
        <div class="badge-row">
            <span class="badge">Accuracy: {ACCURACY * 100:.2f}%</span>
            <span class="badge">784 ➔ 128 ➔ 64 ➔ 10 MLP</span>
            <span class="badge">Zero Server Cost</span>
            <span class="badge">Instant Client-side Inference</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("✍️ Draw Digit"):
                    sketch_input = gr.Sketchpad(
                        label="Draw a digit (0 to 9)",
                        type="numpy",
                        image_mode="RGBA",
                        height=280
                    )
                    with gr.Row():
                        btn_clear_draw = gr.Button("🧹 Clear", size="sm")
                        btn_pred_draw = gr.Button("🚀 Classify Drawing", variant="primary", size="lg")

                with gr.Tab("📁 Upload Image"):
                    upload_input = gr.Image(
                        label="Upload digit image (PNG, JPG, JPEG)",
                        type="numpy",
                        image_mode="RGBA"
                    )
                    btn_pred_upload = gr.Button("🚀 Classify Upload", variant="primary", size="lg")

                with gr.Tab("🎲 Test MNIST Samples"):
                    gr.Markdown("Select test samples directly from the standard MNIST dataset:")
                    with gr.Row():
                        digit_selector = gr.Dropdown(
                            choices=[str(i) for i in range(10)],
                            value="3",
                            label="Target Digit Class"
                        )
                        variant_selector = gr.Radio(
                            choices=["0", "1", "2", "3"],
                            value="0",
                            label="Variant #"
                        )
                    sample_preview = gr.Image(
                        value=load_sample_digit("3", "0"),
                        label="Selected Test Digit (28x28)",
                        type="pil",
                        height=200
                    )
                    btn_pred_sample = gr.Button("🚀 Classify Sample", variant="primary", size="lg")

        with gr.Column(scale=4):
            gr.Markdown("### 📊 Prediction Results")
            result_label = gr.Markdown("👆 Draw, upload, or choose a sample to see predictions.")
            preview_28x28 = gr.Image(label="Model Input (28x28 Resized)", type="pil", height=120)
            confidence_chart = gr.Label(num_top_classes=10, label="Probability Distribution (0-9)")

    # Event handlers
    btn_pred_draw.click(
        fn=predict_digit,
        inputs=[sketch_input],
        outputs=[result_label, confidence_chart, preview_28x28]
    )
    
    btn_pred_upload.click(
        fn=predict_digit,
        inputs=[upload_input],
        outputs=[result_label, confidence_chart, preview_28x28]
    )

    def on_sample_change(d, v):
        return load_sample_digit(d, v)

    digit_selector.change(on_sample_change, inputs=[digit_selector, variant_selector], outputs=[sample_preview])
    variant_selector.change(on_sample_change, inputs=[digit_selector, variant_selector], outputs=[sample_preview])

    btn_pred_sample.click(
        fn=predict_digit,
        inputs=[sample_preview],
        outputs=[result_label, confidence_chart, preview_28x28]
    )

    btn_clear_draw.click(lambda: None, None, sketch_input)

if __name__ == "__main__":
    demo.launch()
