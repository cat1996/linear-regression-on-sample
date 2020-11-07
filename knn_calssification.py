# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:35:03 2020

@author: hemchaitanya
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import math
import operator
#sorting data
def loaddataset(filename,split,training=[],test = []):
    with open(filename,'r') as csvfile:     #readign data file from csv file   
        lines = csv.reader(csvfile)
        dataset = list(lines)            #converting csvread lines to list
        for x in range(1,len(dataset)-1):   #for loop
            for y in range(1,3):
                dataset[x][y] = float(dataset[x][y])
                if random.random() <split:   #splitting data into 70,30 percent
                    training.append(dataset[x])
                else:
                    test.append(dataset[x])    #stroing training and testing data separetly
def ecsdist(i1,i2,leng):    #fuction for calculating eculidoean distance
    dist = 0  #initilizing distance to zero
    for x in range(1,leng):
        dist+=pow((i1[x]-i2[x]),2) #calculating square of distance by using for loop
    return math.sqrt(dist)  #returning square root of distance
def getneigh(trainingset,i1,k):   #fuction for gettting neighbour values for our k
    dist = []            #empty list for dist fuction
    l1 = len(i1)-1   #calculating lenght for i1 list
    for x in range(len(trainingset)-1):   # for loop for calcutlating distance for every training value
        dist1 = ecsdist(i1,trainingset[x],l1) #distacne calculated for each and every training value
        dist.append((trainingset[x],dist1))  #distance is appended ti dist list
    dist.sort(key = operator.itemgetter(1))  # sorting dist function
    neighbours = []  #declaring empty list to store k neighbours
    for x in range(k):   # for loop for stroring fissr k neighbours
        neighbours.append(dist[x][0])
    return neighbours
def getresp(neighbours): # fuction for effectively getting response uing height vlaue and weight value
    vote = {}  #creating empty distnary for both recording n of males and females 
    for x in range(len(neighbours)):
        response = neighbours[x][0] #checking wehter response is male or female
        if response in vote: # if response is vote then 1 is added
            vote[response]+=1
        else: # else the vote[response] is 1
            vote[response]= 1
    sortedvotes = sorted(vote.items(),key = operator.itemgetter(1),reverse = True)#sorting votes based on items
    return sortedvotes[0][0] #returning respone
"""function for calculating accuracy"""
def getaccu(test,predict):
    correct = 0  #declaringcorrect value as zero
    for x in range(len(test)): 
        if test[x][0] == predict[x]: # chcking whether our prediceted values is coreect or not
            correct+=1 #if it is correct then 1 is added to correct
    p = (correct/float(len(test)))*100  #caluclating percentage of correct with test values
    return p
"""function for confusion matrix"""
def cf(predict,test):
    cm = {'mm':0,'mf':0,'fm':0,'ff':0}  #declaring empty confusion matrix
    for x in range(len(test)):
        if predict[x] == 'Male':    #if prdicted valus is male
            if test[x][0] == predict[x]: #if predicted alue is correct
                cm['mm']+=1 #then 1 is added to cm[mm] implying prdicted and test value is male
            else:  
                cm['mf']+=1  #else 1 is added cm[mf] implying predicted male when correct answer is female 
        else:#similary for female also
            if test[x][0] == predict[x]:
                cm['ff']+=1
            else:
                cm['fm']+=1
    return cm

            
    
training = []  #empty training list
test =[]   #empty test lis
fem = []
heim = []#empty height list for male
weim = []  #emty weight lis for male
heif = []
weif = []
loaddataset(r"weight-height.csv",0.7,training,test) #function for sorting lis tbased for training adn testing 
print('train:'+repr(len(training)))
print('test:'+repr(len(test)))
t1=ecsdist(training[1],training[100],3)#claculating ecludian distance wheb
for x in range(len(training)):   #separating taringing data based on female and male
    if (training[x][0] == 'Female'):
        fem.append(training[x])
        weif.append(training[x][1])
        heif.append(training[x][2])
    else:
        weim.append(training[x][1])
        heim.append(training[x][2])
plt.scatter(weim,heim,c = "blue")   #plotiing scatter plit for male and female
plt.scatter(weif,heif,c = "red")
plt.xlabel("height")
plt.ylabel("weight")
k =math.sqrt(7000)    #optimum k value
k = int(k)
predict = []
for x in range(len(test)):    #getting respones for every test value 
    neighbours = getneigh(training,test[x],k)
    response = getresp(neighbours)
    predict.append(response)
acc = getaccu(test,predict)   #calculating accuracy
print("the accuracy is "+str(acc))
cm = cf(predict,test)   #calculating confusion matrix
recall_male = cm['mm']/(cm['mm']+cm['mf'])  #recall score for male
print("the recall score for male is "+str(recall_male))
recall_fmale = cm['ff']/(cm['ff']+cm['fm'])
print("the recall score for female is"+str(recall_fmale))
f_scm = (2*acc*recall_male)/(acc+recall_male)  #fcsore for male and femae
f_scf = (2*acc*recall_fmale)/(acc+recall_fmale)
print("the f score for male is "+str(f_scm)+" the f score for female is "+str(f_scf))
print('The confusion matrix is',cm)

