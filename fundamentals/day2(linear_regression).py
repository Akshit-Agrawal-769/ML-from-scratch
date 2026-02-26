import numpy as np
import matplotlib.pyplot as plt

# X = np.array([1,2,3])
# Y = np.array([2,4,7])
# Xm = np.mean(X)
# Ym = np.mean(Y)
# w = (np.dot((X-Xm),(Y-Ym))/np.dot((X-Xm),(X-Xm)))
# b = Ym - w*Xm

# plt.scatter(X,Y)
# plt.plot(X,w*X+b)
# plt.show()

def generate_Data(w, b, std, n=100):
    X = np.linspace(-5,5,n)
    noise = np.random.normal(0, std, n)
    Y = w * X + b + noise
    return X, Y

def closed_Form(X, Y):
    Xm = np.mean(X)
    Ym = np.mean(Y)
    w = (np.dot( (X - Xm), (Y - Ym) ) / np.dot( (X - Xm), (X - Xm) ))
    b = Ym - w*Xm
    return w, b

X1, Y1 = generate_Data(3, 2, std = 0.5)
w1, b1 = closed_Form(X1, Y1)

print('low noise slope',w1)
plt.scatter(X1, Y1)
plt.plot(X1, w1 * X1 + b1)
plt.show()

X2, Y2 = generate_Data(3, 2, std = 5)
w2, b2 = closed_Form(X2, Y2)

print('high noise slope',w2)    
plt.scatter(X2, Y2)
plt.plot(X2, w2 * X2 + b2)
plt.show()

def mse(Y,fit_Y):
    residual = Y - fit_Y
    return np.mean((residual)**2)

print('low noise mse',mse(Y1,w1*X1+b1)) 
print('high noise mse',mse(Y2,w2*X2+b2)) 

slopes = []

for i in range(1000):
    X, Y = generate_Data(3, 2, std=5)
    w, b = closed_Form(X, Y)
    slopes.append(w)

print("Mean slope:", np.mean(slopes))
print("Std of slope:", np.std(slopes))

slopes2=[]

for i in range(1000):
    X, Y = generate_Data(3, 2, std=0.5)
    w, b = closed_Form(X, Y)
    slopes2.append(w)

print("Mean slope:", np.mean(slopes2))
print("Std of slope:", np.std(slopes2))