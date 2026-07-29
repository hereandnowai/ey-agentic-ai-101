import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging (1)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN to avoid AVX/FMA warnings

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

import absl.logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)  # Suppress absl logging

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logging (2)
np.random.seed(42)
tf.random.set_seed(42)

def sigmoid(x):                     # signmoid squashes any number into a value between 0 and 1.
    return 1 / (1 + np.exp(-x))


# part 1 - one neuron, built by hand
FEATURES = ["income", "low_debt", "credit_history"]
inputs = np.array([0.8, 0.3, 0.9])  # Example input features
weights = np.array([0.5, 0.2, 0.3])
bias = -0.4

print("\n--- Part 1: One Neuron Built by Hand ---")
for name, value, weight in zip(FEATURES, inputs, weights):
    print(f"{name:<16}{value:.1f} x {weight:.1f} = {value * weight:.2f}")

total = np.dot(inputs, weights) + bias
print(f"\nTotal (weighted sum + bias): {total:.3f}")
print(f" after squashing with sigmoid: {sigmoid(total):.3f}")
print(f" >> {sigmoid(total):.0%} chance this loan is good")

# part 2 - the activation and why a netword is useless without it
print("\n--- Part 2: Activation Function ---")
print(f" {'RAW':<9}{'SIGMOID':<11}{'RELU':<9}READING")
for raw in [-4.0, -1.0, 0.0, 1.0, 4.0]:
    s = sigmoid(raw)
    reading = ("confident no" if s < 0.2 else
               "confident yes" if s > 0.8 else "unsure")
    print(f" {raw:<9.1f}{s:<11.3f}{max(0.0, raw):<9.1f}{reading}")


# part 3 - watching a neuron learn
X = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3])
y = np.array([1, 1, 1, 0, 0, 0])  # Labels: 1 for good loan, 0 for bad loan
w, b = 0.0, 0.0  # Initialize weights and bias
learning_rate = 2.0

print("\n--- Part 3: Watching a Neuron Learn (starts at w=0.0, b=0.0) ---")
print(f" {'ROUND':<8}{'WEIGHT':<9}{'BIAS':<7}{'LOSS':<7}")
for step in range(301):
    predictions = sigmoid(X * w + b)
    error = predictions - y
    if step % 60 == 0:
        loss = np.mean(error ** 2)
        print(f" {step:<8}{w:<10.3f}{b:<10.3f}{loss:.4f}")
    w -= learning_rate * np.mean(error * X)
    b -= learning_rate * np.mean(error)

print(f" final: a high income now scores {sigmoid(0.9 * w + b):.0%} chance of a good loan"
      f" a low one {sigmoid(0.1 * w + b):.0%} chance")
