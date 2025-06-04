import torch
import matplotlib.pyplot as plt
from model import CIRCLENet

model = CIRCLENet()
model.load_state_dict(torch.load("saved_model/circle_model.pth"))
model.eval()

n_points = 1000
X_test = torch.rand(n_points, 2) * 3 - 1.5
Y_test = ((X_test[:, 0]**2 + X_test[:, 1]**2) < 1).float().unsqueeze(1)

with torch.no_grad():
    Y_pred = model(X_test)
    Y_pred_bin = (Y_pred > 0.5).float()

accuracy = (Y_pred_bin == Y_test).float().mean().item()
print(f"Model accuracy on random circle data: {accuracy * 100:.2f}%")

X_np = X_test.numpy()
Y_pred_np = Y_pred_bin.numpy().flatten()

plt.figure(figsize=(6, 6))
plt.scatter(X_np[:, 0], X_np[:, 1], c=Y_pred_np, cmap="coolwarm", s=10)
plt.title("Model Predictions: Inside Circle Classification")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()
