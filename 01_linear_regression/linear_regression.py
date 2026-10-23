import numpy as np
# feature
X = np.array([30,40,50,60,70])

# parameter
W = 20
b = 100

# hyperparameter
learning_rate = 0.00001
epochs = 1000

# reality
y_true = np.array([900,1200,1500,1800,2100])

for epoch in range(epochs):
    # prediction
    y_pred = W * X + b

    # errors
    error = y_pred - y_true

    # gradients
    dW = (2 / len(X)) * np.sum(X * error)
    db = (2 / len(X)) * np.sum(error)

    # update parameters
    W = W - learning_rate * dW
    b = b - learning_rate * db

    # print progress
    if (epoch + 1) % 100 == 0:
        y_pred = W * X + b
        mse = np.mean((y_pred - y_true) ** 2)

        print(
            f"Epoch {epoch + 1}: "
            f"W={W:.4f}, "
            f"b={b:.4f}, "
            f"MSE={mse:.4f}"
        )

# 5. Final prediction
y_pred = W * X + b

print("\nFinal W:", W)
print("Final b:", b)
print("Final MSE:", np.mean((y_pred - y_true) ** 2))


