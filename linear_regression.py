# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:59:43 2020

@author: hemchaitanya
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import csv
def loaddataset(filename,split,training=[],test = []):
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset)-1):
            for y in range(1,3):
                dataset[x][y] = float(dataset[x][y])
                if random.random() <split:
                    training.append(dataset[x])
                else:
                    test.append(dataset[x])


"""linear model"""
def ln(X,w):
    return(w[1]*np.array(X[:,0])+w[0])
"""cost function"""
def costfun(w,X,y):
    return (0.5/m) *np.sum(np.square(ln(X,w))-np.array(y))
"""grad desecent"""
def grad(w,X,y):
    g = [0,0]
    g[0] = (1/m)*np.sum(ln(X,w)-np.array(y))
    g[1] = ((1/m)*np.sum(ln(X,w)-np.array(y))*np.array(X[:,0]))
    return g
"""gradient decent"""
def descent(w_new, w_prev, lr,X,y):
    print(w_prev)
    print(costfun(w_prev,X,y))

    j=0
    p1 =[0,0]
    while True:
        w_prev = w_new
        p1[0] = (1/m)*np.sum(ln(X,w)-np.array(y))
        p1[1] = (1/m) * np.sum((ln(X,w)-np.array(y))*np.array(X[:,0]))

        print(p1)

        w0 = w_prev[0] - lr*p1[0]
        w1 = w_prev[1] - lr*p1[1]
        w_new = [w0, w1]
        print(w_new)
        print(costfun(w_new,X,y))
        t1 = (w_new[0]-w_prev[0])**2+(w_new[1]-w_prev[1]**2)
        print(t1)
        if t1 < pow(10,-6):
            return w_new
        if j>500: 
            return w_new
        j+=1
"""reading data"""
data = pd.read_csv("weight-height.csv")
print(data.shape)
data.head()
wei =[]
hei=[]
wei2 =[]
hei2 =[]
wei_t=[]
hei_t =[]
w_t =[]
h_t = []
training = []
test =[]
loaddataset(r"weight-height.csv",0.7,training,test)
for x in range(0,len(training)):
    w_t.append(training[x][1])
    h_t.append(training[x][2])
wei1 = data["Height"].values
hei1 = data["Weight"].values
"""sorting data"""
for x in range(0,len(test)):
    wei2.append(test[x][1])
    hei2.append(test[x][2])
wei_t = wei2
hei_t = hei2
m = len(w_t)

wei_lr = np.array(w_t)
hei_lr = np.array(h_t)
X = wei_lr.reshape(-1,1)
y = hei_lr
plt.scatter(wei_lr,hei_lr)
plt.xlabel('weight')
plt.ylabel('height')
plt.show()
mean_hei = np.mean(hei_lr)
mean_wei = np.mean(wei_lr)
numer = 0
denom =0
"""calculate b1,b0"""
for i in range(len(wei_lr)):
    numer+=(wei_lr[i]-mean_wei)*(hei_lr[i]-mean_hei)
    denom+=(wei_lr[i]-mean_wei)**2
b1 = numer/denom
b0 = mean_hei-(b1*mean_wei)
w = [b0,b1]
print(b1,b0)
w = descent(w,w,.001,X,y)
max_wei = max(wei_lr)
min_wei = min(wei_lr)
x = np.linspace(min_wei,max_wei,20000)
print(w)
"""plotting graph for linear regression"""
def graph(formula, x_range):  
    x = np.array(x_range)  
    y = formula(x)  
    plt.plot(x, y)  
def graph1(formula, x_range):  
    x = np.array(x_range)  
    y = formula(x)  
    ax = sns.regplot(x, y)  
"""linear regressin actual avlue"""    
def my_formula(x):
    return w[0]+w[1]*x

plt.scatter(X,y, c = "red",alpha=.5, marker = 'o')
graph(my_formula, x)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""r^2 value"""
ss_t =0
ss_r = 0
for i in range(len(wei_t)):
    y_pred =w[0]+w[1]*wei_t[i]
    ss_t+= (hei_t[i]-mean_hei)**2
    ss_r+=(hei_t[i]-y_pred)**2
r2 = 1-(ss_r/ss_t)
print('the R^2 score is '+str(r2))
ax = sns.scatterplot(w_t,h_t)
graph1(my_formula,x)