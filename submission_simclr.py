import onnxruntime as ort
import numpy as np
import requests
import os

# === CONFIG ===
SEED = "61437433"      
TOKEN = "09596680"    
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR.onnx"                     # MSE_alpha = 0.5           L2 = 6.81
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_07.onnx"                  # MSE_alpha = 0.7           L2 = 6.6186
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_09.onnx"                  # MSE_alpha = 0.9           L2 = 6.6148
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_cosine_norm_loss.onnx"   # cosine_norm_alpha = 0.9    L2 = 19.93
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_cosine.onnx"              # alpha = 0.7 _&_ cosine loss   L2 = ?
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_NT_xent.onnx"             # NT xent _&_ tau = 1.0     L2 = 35.20
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_resnet50.onnx"             # resnet50                    L2 = 22.044


# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_MSE_loss_opt.onnx"             # resnet18, MSE-full optimized  L2 = 6.030
# ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_DINO_100.onnx"                 # DINO                          L2 = ?
ONNX_PATH = "F:\\TML_model_stealing\\TML25_A2_39\\models\\model_Sim_CLR_DINO_150_optim.onnx"           # DINO_optim 150 epochs         L2 = 28.60

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
