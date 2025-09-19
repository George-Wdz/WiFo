import numpy as np
import os
import matplotlib.pyplot as plt

pred = np.load("y_pred_D1_temporal_0.5.npz")
targ = np.load("y_target_D1_temporal_0.5.npz")
meta = np.load("meta_D1_temporal_0.5.npz")

pred_o = np.load("y_pred_decoded_D1_temporal_0.5.npz")
targ_o = np.load("y_target_decoded_D1_temporal_0.5.npz")

print("Pred keys:", pred.files)
print("Targ keys:", targ.files)
print("Meta keys:", meta.files)
print("Pred_o keys:", pred_o.files)
print("Targ_o keys:", targ_o.files)

pred_arr = pred[pred.files[0]]
targ_arr = targ[targ.files[0]]
meta_arr = meta[meta.files[0]]
pred_o_arr = pred_o[pred_o.files[0]]
targ_o_arr = targ_o[targ_o.files[0]]

y_pred = pred_arr[:, :]
y_target = targ_arr[:, :]
patch_info = meta_arr[:]
y_pred_o = pred_o_arr[:, :]
y_target_o = targ_o_arr[:, :]


print("Pred shape:", pred_arr.shape, "range:", pred_arr.min(), pred_arr.max())
print("Targ shape:", targ_arr.shape, "range:", targ_arr.min(), targ_arr.max())
print("Meta shape:", meta_arr.shape, "range:", meta_arr.min(), meta_arr.max())
print("Pred_o shape:", pred_o_arr.shape, "range:", pred_o_arr.min(), pred_o_arr.max())
print("Targ_o shape:", targ_o_arr.shape, "range:", targ_o_arr.min(), targ_o_arr.max())

import torch

def get_compute_device():
    """Detect and return the best available compute device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (no GPU available)")
    return device

def normalized_mean_squared_error(y_true, y_pred, device=None):
    """
    Calculate NMSE for complex CSI data using Frobenius norm:
    NMSE = ||H_pred - H_true||_F^2 / ||H_true||_F^2
    
    Args:
        y_true: numpy array or torch tensor of true CSI (can be complex)
        y_pred: numpy array or torch tensor of predicted CSI
        device: torch device (cpu/cuda), auto-detected if None
        
    Returns:
        nmse: Normalized Mean Squared Error
    """
    # Convert inputs to PyTorch tensors if needed
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.from_numpy(np.asarray(y_true))
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.from_numpy(np.asarray(y_pred))
    
    # Auto-detect device if not specified
    if device is None:
        device = get_compute_device()
    
    # Move tensors to device and ensure complex dtype if needed
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    
    # Validate shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    
    # Calculate Frobenius norms
    diff = y_true - y_pred
    numerator = torch.linalg.norm(diff, ord='fro') ** 2
    denominator = torch.linalg.norm(y_true, ord='fro') ** 2
    
    # Handle zero denominator case
    if denominator == 0:
        print("Warning: Denominator is zero - returning numerator")
        return numerator.item()
    
    return (numerator / denominator).item()

# Calculate and display NMSE results
nmse = normalized_mean_squared_error(y_target, y_pred)

print("\n=== Enhanced NMSE Analysis ===")
print(f"Data points: {len(y_target.flatten())}")
print(f"Target range: [{y_target.min():.4f}, {y_target.max():.4f}]")
print(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"MSE: {np.mean((y_target - y_pred)**2):.6f}")
print(f"Target variance: {np.var(y_target):.6f}")
print(f"NMSE: {nmse:.6f}")

