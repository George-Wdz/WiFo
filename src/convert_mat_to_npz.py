# coding=utf-8
import os
import numpy as np
import scipy.io

def convert_mat_to_npz(input_path, output_path):
    """
    Convert .mat file to .npz format.

    Args:
        input_path (str): Path to the input .mat file.
        output_path (str): Path to save the output .npz file.
    """
    # Load .mat file
    mat_data = scipy.io.loadmat(input_path)

    # Extract the variable 'X_val'
    if 'X_val' not in mat_data:
        raise KeyError("The .mat file does not contain the variable 'X_val'.")

    X_val = mat_data['X_val']  # Shape: [samples, time_blocks, antennas, freq_blocks]

    # Check if the data is complex
    if not np.iscomplexobj(X_val):
        raise ValueError("The variable 'X_val' is not complex.")

    # Split into real and imaginary parts and concatenate along a new channel dimension
    X_real = np.real(X_val)
    X_imag = np.imag(X_val)
    X_combined = np.stack((X_real, X_imag), axis=1)  # Shape: [samples, 2, time_blocks, antennas, freq_blocks]

    # Save as .npz file
    np.savez(output_path, X_combined=X_combined)

    print(f"Converted {input_path} to {output_path}")

if __name__ == "__main__":
    # Example usage
    dataset_name = "D1"  # Change this to the desired dataset name
    input_file = f"../dataset/{dataset_name}/X_test.mat"
    output_file = f"../dataset/{dataset_name}/{dataset_name}_converted.npz"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Perform conversion
    convert_mat_to_npz(input_file, output_file)