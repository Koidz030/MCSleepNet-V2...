#! /usr/bin/python
# -*- coding: utf-8 -*-​

import numpy as np
import pickle as pk
import random as rn
import csv

# seed random number generator
np.random.seed(3360)
rn.seed(840)
id4200=[]

# Load the whole mouse ID list of large dataset (4200 mice)
with open('/data1/mouse_4313/datalist_4200.csv', 'r') as f:
    reader=csv.reader(f)
    for raw in reader:
        id4200.extend(raw)
        
# Random datalist from the 4200 mice (320 mice training)
trainlist1 = rn.sample(id4200,320)
trainlist1=np.ravel(trainlist1)

# Deleting trainlist1 to avoid repeating same samples in testlist1
deleted_id = [np.where(id4200==sample) for sample in trainlist1]
deleted_id = np.ravel(deleted_id)
testlist1 = np.delete(id4200, deleted_id)


# New train_subset lists with randomly sampled data (16 mice)
new_train_subset = {}
for i in range (20):
    new_train_subset[i] = rn.sample(trainlist1,16)


# New test_subset lists with randomly sampled data (160 mice)
testlist1 = rn.sample(id4200, 160)
testlist1=np.ravel(testlist1)


#saving the first trainlist
with open('trainlist1.csv', 'w') as f:
    writer=csv.writer(f)
    writer.writerow(trainlist1)

#saving the first testlist
with open('testlist1.csv', 'w') as f:
    writer=csv.writer(f)
    writer.writerow(testlist1)


#saving the subsequent trainlists
with open('train_subsets.pkl', 'wb')  as f:
   pk.dump(new_train_subset, f)


