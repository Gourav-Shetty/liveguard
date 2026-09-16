import os
import numpy as np
import torch

def export_model_to_numpy(pth_path, out_npz_path):
    state = torch.load(pth_path, map_location="cpu")
    
    c1_w = state["features.0.weight"].numpy()
    c1_b = state["features.0.bias"].numpy() if "features.0.bias" in state else np.zeros(16, dtype=np.float32)
    bn1_w = state["features.1.weight"].numpy()
    bn1_b = state["features.1.bias"].numpy()
    bn1_m = state["features.1.running_mean"].numpy()
    bn1_v = state["features.1.running_var"].numpy()
    
    scale1 = bn1_w / np.sqrt(bn1_v + 1e-5)
    fused_c1_w = c1_w * scale1[:, None, None]
    fused_c1_b = (c1_b - bn1_m) * scale1 + bn1_b

    c2_w = state["features.4.weight"].numpy()
    c2_b = state["features.4.bias"].numpy() if "features.4.bias" in state else np.zeros(32, dtype=np.float32)
    bn2_w = state["features.5.weight"].numpy()
    bn2_b = state["features.5.bias"].numpy()
    bn2_m = state["features.5.running_mean"].numpy()
    bn2_v = state["features.5.running_var"].numpy()

    scale2 = bn2_w / np.sqrt(bn2_v + 1e-5)
    fused_c2_w = c2_w * scale2[:, None, None]
    fused_c2_b = (c2_b - bn2_m) * scale2 + bn2_b

    fc1_w = state["classifier.1.weight"].numpy()
    fc1_b = state["classifier.1.bias"].numpy()
    fc2_w = state["classifier.4.weight"].numpy()
    fc2_b = state["classifier.4.bias"].numpy()

    np.savez_compressed(
        out_npz_path,
        c1_w=fused_c1_w,
        c1_b=fused_c1_b,
        c2_w=fused_c2_w,
        c2_b=fused_c2_b,
        fc1_w=fc1_w,
        fc1_b=fc1_b,
        fc2_w=fc2_w,
        fc2_b=fc2_b
    )
    print(f"Exported fused weights to: {out_npz_path} ({os.path.getsize(out_npz_path)} bytes)")

if __name__ == "__main__":
    pth = "quickstart-pytorch/stage1_cnn.pth"
    out = "quickstart-pytorch/stage1_weights.npz"
    if os.path.exists(pth):
        export_model_to_numpy(pth, out)
    else:
        print(f"{pth} not found yet. Run after training finishes.")
