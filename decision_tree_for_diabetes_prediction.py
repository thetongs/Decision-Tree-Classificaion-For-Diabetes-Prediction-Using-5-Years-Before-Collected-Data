#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Dataset Information
# Information about 768 people, whether they were diagnosed with diabetes within five years after the information was recorded.


# In[5]:


## Load dataset
#
with open("diabetes.csv", "r") as f:
    data = f.readlines()

data[:10]


# In[10]:


## Need to format data into our required format
# All features set 
feats = data[0]   # Name of all columns 
feats = feats.replace('\n', '')
feats = feats.split(",")

# Independent feature set names
feats = feats[0:(len(feats) - 1)]

# Doing same for all dataset
dat = []
labs = []

for i in range(1, len(data)):
    line = data[i]
    line = line.replace('\n', '')
    csvline = line.split(",")
    labs = labs + [int(csvline[len(csvline) - 1])]
    csvline = [float(csvline[i]) for i in range(len(csvline) - 1)]
    dat = dat + [csvline]

print(labs[:5])
print(dat[:5])


# In[15]:


## Decision tree
# 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_leaf_nodes = 3)
classifier = classifier.fit(dat, labs)

# Check model results
correct = 0
for i in range(len(dat)):
    if classifier.predict([dat[i]]) == labs[i]:
        correct = correct + 1

print("Percentage of correct : {} %".format((correct / len(dat)) * 100))


# In[16]:


get_ipython().system(' jupyter nbconvert --to script decision_tree_for_diabetes_prediction.ipynb')

