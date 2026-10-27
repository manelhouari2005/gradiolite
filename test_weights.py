import json
import zlib
import base64
import numpy as np
from model_weights_b64 import COMPRESSED_DATA

raw_json = zlib.decompress(base64.b64decode(COMPRESSED_DATA)).decode('utf-8')
data = json.loads(raw_json)

W1 = np.array(data['w1'], dtype=np.float32)
B1 = np.array(data['b1'], dtype=np.float32)
W2 = np.array(data['w2'], dtype=np.float32)
B2 = np.array(data['b2'], dtype=np.float32)
W3 = np.array(data['w3'], dtype=np.float32)
B3 = np.array(data['b3'], dtype=np.float32)

print("W1 shape:", W1.shape, "B1 shape:", B1.shape)
print("Accuracy:", data['acc'])
print("Sample keys:", list(data['samples'].keys()))

# Test sample digit 7
sample_7 = np.array(data['samples']['7'][0], dtype=np.float32).reshape(1, -1)
h1 = np.maximum(0, np.dot(sample_7, W1) + B1)
h2 = np.maximum(0, np.dot(h1, W2) + B2)
logits = np.dot(h2, W3) + B3
exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
pred = int(np.argmax(probs))
print(f"Predicted class for digit 7 sample: {pred} with probability {probs[0][pred]:.4f}")
assert pred == 7
print("Verification passed successfully!")
