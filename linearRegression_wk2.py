# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:32:18 2018

@author: Immanuel
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Immanuel/Downloads/lifeexpectancy_usa.csv")

print(data.shape)

print (data.head())
print (data.tail())

data.describe()

X = data ['TIME'].values

print(X)

Y = data ['Value'].values

print(Y)

mean_x = np.mean(X)
mean_y = np.mean(Y)

print(mean_x)
print(mean_y)

m= len(X)

numer = 0
denom = 0

for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    
    denom += (X[i] - mean_x) **2
    b1 = numer / denom
    b0 = mean_y - (b1 * mean_x)
    
    print(b1, b0)
    
max_x  = np.max(X) + 10
    
min_x = np.min(X) - 10
    
print(max_x)
print(min_x)
    
x = np.linspace(min_x, max_x, 10)
y = b0 + b1 * x
    
print(x)
    
plt.plot(x, y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Period in Years')
plt.ylabel('Value in Years')

plt.legend()
plt.show()

max_x = np.max(X) + 10
min_x = np.min(X) - 10

x = np.linspace(min_x, max_x, 10)
y = b0 + b1 * x

plt.plot(x, y, color='#b95871', label='Regression Line')
plt.scatter(X, Y, c='#5871b9', label='Scatter Plot')

plt.xlabel('period years')

plt.ylabel('Value in Years')
plt.legend()
plt.show()

rmse = 0

for i in range(m):
    y_pred = b0 = b1 * X[i]
    rmse += (Y[1] - y_pred) ** 2
rmse = np.sqrt(rmse/m)
print(rmse)

ss_t = 0
ss_r = 0

for i in range(m):
    y_pred = b0 + b1 * X[1]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2

r2 = 1 - (ss_r/ss_t)

print(r2)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((m, 1))

reg = LinearRegression()

reg = reg>fit(X, Y)

Y_pred = reg>predict(X, Y)

mse = mean_squared_error(Y, Y_pred)

rmse = np.sqrt(mse)


r2_score = reg.score(X, Y)

print(np.sqrt(mse))
print(r2_score)
    