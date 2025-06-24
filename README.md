# TML25_A2_39

# TML Assignment 2 – Model Stealing under B4B

**Team 39** • Token: 09596680  
Soham Ritesh Tolwala – soto00002@stud.uni-saarland.de  
Laxmiraman Dnyaneshwar Gudewar – lagu00003@stud.uni-saarland.de  

---
## Folder Map

| Path | What’s inside |
|------|---------------|
| `data/ModelStealingPub.pt` | Public image set (1 000 × 32 × 32 RGB) |
| `extract/api_client.py` | Thin wrapper around the victim REST API; caches JSON<br>→ creates `reps.npy` (1024-D victim embeddings) |
| `extract/extract_embeddings.ipynb` | One-shot notebook that calls the API *once* per image and stores embeddings |
| `train/train_ssl.ipynb` | Main training notebook – SimCLR views, EMA teacher, α grid-search |
| `models/` | Instead of huge models, this folder contains names of variant models used |
| `results_graphs/` | Loss curves for every major run (PNG) |
| `submission_simclr.py` | ONNX integrity check + HTTP POST to leaderboard |

---

## Quick Start

```bash
# 0. env (optional)
conda env create -f environment.yml
conda activate tml-steal

# 1. query victim (one time)
python extract/api_client.py      # creates extract/reps.npy

# 2. train & export best model
python train/train_ssl_with_procrustes.py
# -> outputs models/best_simclr_procrustes.onnx

# 3. submit
python submit/submission_simclr.py models/best_simclr_procrustes.onnx
```
---
## Approaches tried:

| Model File Name                           | Method Description                                   | L₂ Distance       |
| ----------------------------------------- | ---------------------------------------------------- | ----------------- |
| `model_Sim_CLR.onnx`                      | SimCLR-style, α = 0.5                                | 6.81              |
| `model_Sim_CLR_07.onnx`                   | SimCLR-style, α = 0.7                                | 6.6186            |
| `model_Sim_CLR_09.onnx`                   | SimCLR-style, α = 0.9                                | 6.6148            |
| **`model_Sim_CLR_MSE_loss_opt.onnx`**     | **Optimized SimCLR + scheduler (ResNet18 backbone)** | **6.0300 ← best** |
| `model_Sim_CLR_Procrustes.onnx`           | Post-hoc Procrustes alignment                        | 6.41              |
| `model_Sim_CLR_Procrustes200.onnx`        | Procrustes, 200 epochs, α ↑ to 0.98                  | 6.22              |
| `model_Sim_CLR_whitening_procrustes.onnx` | Added global whitening / covariance alignment        | 6.24              |
| `model_Sim_CLR_cosine_norm_loss.onnx`     | Cosine-norm loss, α = 0.9                            | 19.93             |
| `model_Sim_CLR_cosine.onnx`               | Cosine loss, α = 0.7                                 | *Not tested*      |
| `model_Sim_CLR_NT_xent.onnx`              | NT-Xent, τ = 1.0                                     | 35.20             |
| `model_Sim_CLR_resnet50.onnx`             | Weighted MSE (α tuned), ResNet50 backbone            | 22.044            |
| `model_Sim_CLR_DINO_100.onnx`             | DINO-style SSL (100 epochs)                          | *Not tested*      |
| `model_Sim_CLR_DINO_150_optim.onnx`       | DINO-style SSL, optimised (150 epochs)               | 28.60             |


---


Best model training vs epoch plot
[SimCLR300_MSE_optim](/MSE_300optim.png)
