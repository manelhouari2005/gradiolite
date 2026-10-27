import json
import zlib
import base64
import os

with open('model/model_data.json', 'rb') as f:
    raw = f.read()

comp = zlib.compress(raw, level=9)
b64 = base64.b64encode(comp).decode('ascii')

with open('model_weights_b64.py', 'w', encoding='utf-8') as f:
    f.write(f'COMPRESSED_DATA = """{b64}"""\n')

print(f"Successfully generated model_weights_b64.py ({os.path.getsize('model_weights_b64.py') / 1024:.1f} KB)")
