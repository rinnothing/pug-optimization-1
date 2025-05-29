import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as torch_optim

def load_and_normalize(file_path):
    df = pd.read_csv(file_path)
    x = df['time'].values.astype(np.float32)
    y = df['el_power'].values.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return x.reshape(-1, 1), y.reshape(-1, 1)

x_data, y_data = load_and_normalize('ex_1.csv')

def train_keras_model(optimizer, name):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(x_data, y_data, epochs=100, verbose=0)
    pred = model.predict(x_data, verbose=0)
    loss = np.mean((pred.flatten() - y_data.flatten()) ** 2)
    print(f"Keras  ({name}): Loss = {loss:.6f}")

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

def train_torch_model(optimizer_fn, name):
    model = TorchModel()
    optimizer = optimizer_fn(model.parameters())
    criterion = nn.MSELoss()
    x = torch.from_numpy(x_data)
    y = torch.from_numpy(y_data)

    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    print(f"PyTorch({name}): Loss = {loss.item():.6f}")

keras_optimizers = {
    "SGD": tf.keras.optimizers.SGD(0.01),
    "Momentum": tf.keras.optimizers.SGD(0.01, momentum=0.9),
    "Nesterov": tf.keras.optimizers.SGD(0.01, momentum=0.9, nesterov=True),
    "Adagrad": tf.keras.optimizers.Adagrad(0.01),
    "RMSProp": tf.keras.optimizers.RMSprop(0.01),
    "Adam": tf.keras.optimizers.Adam(0.01)
}

torch_optimizers = {
    "SGD": lambda p: torch_optim.SGD(p, lr=0.01),
    "Momentum": lambda p: torch_optim.SGD(p, lr=0.01, momentum=0.9),
    "Nesterov": lambda p: torch_optim.SGD(p, lr=0.01, momentum=0.9, nesterov=True),
    "Adagrad": lambda p: torch_optim.Adagrad(p, lr=0.01),
    "RMSProp": lambda p: torch_optim.RMSprop(p, lr=0.01),
    "Adam": lambda p: torch_optim.Adam(p, lr=0.01)
}

print("=== KERAS ===")
for name, opt in keras_optimizers.items():
    train_keras_model(opt, name)

print("\n=== TORCH ===")
for name, opt_fn in torch_optimizers.items():
    train_torch_model(opt_fn, name)
