import numpy as np
import matplotlib.pyplot as plt

def generate_Data(wx,wy,b,std,n=500):
    noise = np.random.normal(0,std,n)
    #X = np.linspace(-10,10,n) #perfectly correlated, multicollinearity case
    #Y = np.linspace(-5,5,n)
    X = np.random.uniform(-10,10,n)
    Y = np.random.uniform(-5,5,n)
    Z = wx*X + wy*Y + b + noise
    return X,Y,Z
 
def gradient(X,Y,Z,wx,wy,b):
    Z_fit = wx*X + wy*Y + b
    residual = Z - Z_fit
    dwx = -2*np.mean(X*residual)
    dwy = -2*np.mean(Y*residual)
    db = -2*np.mean(residual)
    return dwx,dwy,db,residual

def grad_Descent(X,Y,Z,lr=0.001):
    losses=[]
    wx_ass = 0
    wy_ass = 0
    b_ass = 0
    for i in range(1000):
        dwx,dwy,db,residual = gradient(X,Y,Z,wx_ass,wy_ass,b_ass)
        wx_ass -= dwx*lr
        wy_ass -= dwy*lr*2
        b_ass -= db*lr
        losses.append(np.mean(residual**2))
    return wx_ass,wy_ass,b_ass,losses

X1,Y1,Z1 = generate_Data(5,4,2,5)
wx,wy,b,loss=grad_Descent(X1,Y1,Z1)

print(wx,wy,b)
plt.plot(loss)
plt.title("Loss over iterations")
plt.show()