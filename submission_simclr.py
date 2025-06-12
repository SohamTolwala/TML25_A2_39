import onnxruntime as ort
import numpy as np
import requests
import os

# === CONFIG ===
SEED = "61437433"      
TOKEN = "09596680"    
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR.onnx"        # alpha = 0.5
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_07.onnx"          # alpha = 0.7
ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_09.onnx"          # alpha = 0.9
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_cosine.onnx"          # alpha = 0.7 _&_ cosine loss

# === Validate ONNX Model ===
print("\U0001F50D Testing ONNX format...")
with open(ONNX_PATH, "rb") as f:
    model_data = f.read()
    try:
        session = ort.InferenceSession(model_data)
        dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
        output = session.run(None, {"x": dummy_input})[0]
        print("\u2705 ONNX model valid | Output shape:", output.shape)
        assert output.shape == (1, 1024), "\u274C Output shape must be (1, 1024)"
    except Exception as e:
        raise RuntimeError("\u274C Failed ONNX check:", e)

# === Submit to Leaderboard ===
print("\U0001F4E4 Submitting to leaderboard...")
url = "http://34.122.51.94:9090/stealing"
headers = {"token": TOKEN, "seed": SEED}

with open(ONNX_PATH, "rb") as f:
    response = requests.post(url, files={"file": f}, headers=headers)

if response.status_code == 200:
    print("\U0001F3AF Server Response:", response.json())
else:
    print(f"\u274C Submission failed: {response.status_code}\n{response.text}")
