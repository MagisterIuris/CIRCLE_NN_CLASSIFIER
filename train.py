import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import CIRCLENet

nb_points = 500
X_train = torch.rand(nb_points, 2) * 3 - 1.5
Y_train = ((X_train[:, 0]**2 + X_train[:, 1]**2) < 1).float().unsqueeze(1)

model = CIRCLENet()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.4)

loss_history = []
for epoch in range(100000):
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (BCE)")
plt.grid(True)
plt.show()

with torch.no_grad():
    preds = model(X_train)
    for i in range(4):
        print(f"Input: {X_train[i].tolist()} → Predicted: {preds[i].item():.4f} (Target: {Y_train[i].item()})")

torch.save(model.state_dict(), "saved_model/circle_model.pth")