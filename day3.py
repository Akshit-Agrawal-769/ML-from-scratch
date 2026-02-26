import numpy as np
import matplotlib.pyplot as plt

def generate_Data(w, b, std, n=100):

    X = np.linspace(-5,5,n)
    noise = np.random.normal(0,std,n)
    Y = w*X + b + noise
    return X, Y

def gradient(X, Y, w, b):
   
    Y_fit = w*X + b
    residual = Y - Y_fit
    dw = -2 * np.mean(X*residual)
    db = -2 * np.mean(residual)
    return dw, db

X1, Y1 = generate_Data(3, 2, 0.5)

w = 0
b = 0
lr = 0.001

losses = []

for i in range(1000):
    dw, db = gradient(X1, Y1, w, b)
    w -= lr * dw
    b -= lr * db
    losses.append(np.mean((Y1 - (w*X1 + b))**2))

print("GD slope:", w)
# print("Closed form slope:", w1)
    
plt.plot(losses)
plt.title("Loss over iterations")
plt.show()