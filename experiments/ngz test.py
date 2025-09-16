import numpy as np
import os
import matplotlib.pyplot as plt

pred = np.load("y_pred_D17_temporal_0.5.npz")
targ = np.load("y_target_D17_temporal_0.5.npz")
meta = np.load("meta_D17_temporal_0.5.npz")

print("Pred keys:", pred.files)
print("Targ keys:", targ.files)
print("Meta keys:", meta.files)

pred_arr = pred[pred.files[0]]
targ_arr = targ[targ.files[0]]
meta_arr = meta[meta.files[0]]

data = np.load("predictions_step_0_batch_0.npz")
y_pred = data['y_pred']
y_target = data['y_target']
patch_info = data['patch_info']

print("y_pred shape:", y_pred.shape)
print("y_target shape:", y_target.shape)
print("patch_info:", patch_info)

print("Pred shape:", pred_arr.shape, "range:", pred_arr.min(), pred_arr.max())
print("Targ shape:", targ_arr.shape, "range:", targ_arr.min(), targ_arr.max())

sample_idx = 0
ant_idx = 0   # 选一根天线
pred_sample = pred_arr[sample_idx, :, ant_idx, :]
targ_sample = targ_arr[sample_idx, :, ant_idx, :]

plt.subplot(1,2,1)
plt.imshow(np.abs(pred_sample), aspect="auto")
plt.title("Prediction |ant0|")

plt.subplot(1,2,2)
plt.imshow(np.abs(targ_sample), aspect="auto")
plt.title("Target |ant0|")

plt.show()

error = np.abs(pred_sample - targ_sample)

plt.imshow(error, aspect="auto")
plt.title("Prediction Error (|Pred-Target|)")
plt.colorbar()
plt.show()

subcarrier_idx = 10
plt.plot(np.abs(pred_sample[:, subcarrier_idx]), label="Pred")
plt.plot(np.abs(targ_sample[:, subcarrier_idx]), label="Target")
plt.legend()
plt.title("Amplitude vs Time (subcarrier 10, ant 0)")
plt.show()

# 打印文件夹中所有文件的信息
def print_file_info(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                data = np.load(file_path)
                print(f"File: {file}")
                print("Keys:", data.files)
                for key in data.files:
                    print(f"{key} shape:", data[key].shape)
                print("-")
            except Exception as e:
                print(f"Could not read {file}: {e}")

# 调用函数打印当前目录下所有文件的信息
print_file_info(".")
