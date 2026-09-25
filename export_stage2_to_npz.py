import os
import numpy as np
import torch

def export_stage2_to_numpy(pth_path, out_npz_path):
    state = torch.load(pth_path, map_location="cpu")
    
    # Layer 1: Conv1d(1, 32, 7) + BN(32)
    c1_w = state["features.0.weight"].numpy()
    c1_b = state["features.0.bias"].numpy() if "features.0.bias" in state else np.zeros(32, dtype=np.float32)
    bn1_w = state["features.1.weight"].numpy()
    bn1_b = state["features.1.bias"].numpy()
    bn1_m = state["features.1.running_mean"].numpy()
    bn1_v = state["features.1.running_var"].numpy()
    
    scale1 = bn1_w / np.sqrt(bn1_v + 1e-5)
    fused_c1_w = c1_w * scale1[:, None, None]
    fused_c1_b = (c1_b - bn1_m) * scale1 + bn1_b

    # Layer 2: Conv1d(32, 64, 5) + BN(64)
    c2_w = state["features.4.weight"].numpy()
    c2_b = state["features.4.bias"].numpy() if "features.4.bias" in state else np.zeros(64, dtype=np.float32)
    bn2_w = state["features.5.weight"].numpy()
    bn2_b = state["features.5.bias"].numpy()
    bn2_m = state["features.5.running_mean"].numpy()
    bn2_v = state["features.5.running_var"].numpy()

    scale2 = bn2_w / np.sqrt(bn2_v + 1e-5)
    fused_c2_w = c2_w * scale2[:, None, None]
    fused_c2_b = (c2_b - bn2_m) * scale2 + bn2_b

    # Layer 3: Conv1d(64, 64, 3) + BN(64)
    c3_w = state["features.8.weight"].numpy()
    c3_b = state["features.8.bias"].numpy() if "features.8.bias" in state else np.zeros(64, dtype=np.float32)
    bn3_w = state["features.9.weight"].numpy()
    bn3_b = state["features.9.bias"].numpy()
    bn3_m = state["features.9.running_mean"].numpy()
    bn3_v = state["features.9.running_var"].numpy()

    scale3 = bn3_w / np.sqrt(bn3_v + 1e-5)
    fused_c3_w = c3_w * scale3[:, None, None]
    fused_c3_b = (c3_b - bn3_m) * scale3 + bn3_b

    # Classifier: FC1 (64 -> 32) and FC2 (32 -> 5)
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
        c3_w=fused_c3_w,
        c3_b=fused_c3_b,
        fc1_w=fc1_w,
        fc1_b=fc1_b,
        fc2_w=fc2_w,
        fc2_b=fc2_b
    )
    print(f"Exported Stage 2 fused weights to: {out_npz_path} ({os.path.getsize(out_npz_path)} bytes)")

if __name__ == "__main__":
    pth = "data/stage2_multiclass_cnn.pth"
    if os.path.exists(pth):
        export_stage2_to_numpy(pth, "data/stage2_weights.npz")
        export_stage2_to_numpy(pth, "quickstart-pytorch/stage2_weights.npz")
    else:
        print(f"File not found: {pth}")
