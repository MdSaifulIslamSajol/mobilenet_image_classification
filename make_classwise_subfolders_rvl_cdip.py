# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:31:33 2022

@author: msajol1
"""

import shutil
import pandas as pd
import os
import numpy as np
import csv
# =============================================================================
#  EXTRACTING TRAIN DATA
# =============================================================================
#%%
txtfile = r".\rvl-cdip\labels\train.txt"

csvfile = r".\rvl-cdip\labels\train.csv"
with open(txtfile, 'r') as infile, open(csvfile, 'w') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(",") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)
        
#Loading data
df=pd.read_csv(r".\rvl-cdip\labels\train.csv",header=None)
df.head()

#%%
#naming columns in data
df.columns=['image']

#Splitting the column into 2 columns image and class using space ' ' as delimiter
df = pd.DataFrame(df['image'].str.split(' ',1).tolist(),
                                   columns = ['image','class'])
df.head()
df.shape
#saving that to csv file
df.to_csv(r'.\rvl-cdip\labels\train.csv')
#%%
#changing datatypes fpr easy use
df['image']=df['image'].astype(str)
df['class']=df['class'].astype(int)
#%%
os.chdir("rvl-cdip")
os.mkdir("train")
os.chdir("train")
for i in range(16):
    os.mkdir(str(i))

os.chdir(r"C:\Users\msajol1\DeepDrug\embedding generation\mobilnet rvl cdip")

#Pushin images into corresponding folders based on their class number
for i in range(len(df)):
    for j in range(16):
        if df['class'][i]==j:
            shutil.copy2('rvl-cdip/images/'+df['image'][i],'rvl-cdip/train/'+str(j) +'/')
        else:
            continue
#%%

#Renaming folder names for train set
class_name={'0':'letter','1':'form','2':'email','3':'handwritten','4':'advertisement','5':'scientific report','6':'scientific publication','7':'specification','8':'file folder','9':'news article','10':'budget','11':'invoice','12':'presentation','13':'questionnaire','14':'resume','15':'memo'}
path = r'.\rvl-cdip\train'

i = 0
for j in class_name.keys():
    os.rename(path+'/'+j, path+'/'+class_name[j])
    i=i+1

# =============================================================================
# EXTRACTING TEST DATA
# =============================================================================


txtfile = r"rvl-cdip/labels/test.txt"
csvfile = r"rvl-cdip/labels/test.csv"
with open(txtfile, 'r') as infile, open(csvfile, 'w') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(",") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)
        
#Loading data
test=pd.read_csv(r"rvl-cdip/labels/test.csv",header=None)


#naming columns
test.columns=['image']

#Splitting column into 2 columns using space ' ' as delimiter
df_test = pd.DataFrame(test['image'].str.split(' ',1).tolist(),
                                   columns = ['image','class'])

#chaning datatypes for easy use
df_test['image']=df_test['image'].astype(str)
df_test['class']=df_test['class'].astype(int)


#%%
os.chdir("rvl-cdip")
os.mkdir("test")
os.chdir("test")

for i in range(16):
    os.mkdir(str(i))

os.chdir(r"C:\Users\msajol1\DeepDrug\embedding generation\mobilnet rvl cdip")


#%%
#Pushin images into corresponding folders based on their class number


for i in range(len(df_test)):
    for j in range(16):
        if df_test['class'][i]==j:
            shutil.copy2('rvl-cdip/images/'+df_test['image'][i],'rvl-cdip/test/'+str(j) +'/')
        else:
            continue

#%%
#Renaming folder names for test set
class_name={'0':'letter','1':'form','2':'email','3':'handwritten','4':'advertisement','5':'scientific report','6':'scientific publication','7':'specification','8':'file folder','9':'news article','10':'budget','11':'invoice','12':'presentation','13':'questionnaire','14':'resume','15':'memo'}
path = 'rvl-cdip/test'

i = 0
for j in class_name.keys():
    os.rename(path+'/'+j, path+'/'+class_name[j])
    i=i+1



# =============================================================================
# EXTRACTING VALIDATION DATA
# =============================================================================

txtfile = r"rvl-cdip/labels/val.txt"
csvfile = r"rvl-cdip/labels/validation.csv"
with open(txtfile, 'r') as infile, open(csvfile, 'w') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(",") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)
        
#Loading data
val=pd.read_csv(r'rvl-cdip/labels/validation.csv',header=None)

#Renaming columns
val.columns=['image']

#Splitting column into 2 columns using space ' ' as delimiter
df_val = pd.DataFrame(val['image'].str.split(' ',1).tolist(),
                                   columns = ['image','class'])

#Changing datatypes for eay use
df_val['image']=df_val['image'].astype(str)
df_val['class']=df_val['class'].astype(int)

#Checking data to see if it is done or not
df_val.head()

#%%
os.chdir("rvl-cdip")
os.mkdir("validation")
os.chdir("validation")
for i in range(16):
    os.mkdir(str(i))

os.chdir(r"C:\Users\msajol1\DeepDrug\embedding generation\mobilnet rvl cdip")

#%%

#Pushin images into corresponding folders based on their class number
for i in range(len(df_val)):
    for j in range(16):
        if df_val['class'][i]==j:
            shutil.copy2('rvl-cdip/images/'+df_val['image'][i],'rvl-cdip/validation/'+str(j) +'/')
        else:
            continue
#%%

#Renaming folder names for validation set
class_name={'0':'letter','1':'form','2':'email','3':'handwritten','4':'advertisement','5':'scientific report','6':'scientific publication','7':'specification','8':'file folder','9':'news article','10':'budget','11':'invoice','12':'presentation','13':'questionnaire','14':'resume','15':'memo'}
path = r'rvl-cdip/validation'

i = 0
for j in class_name.keys():
    os.rename(path+'/'+j, path+'/'+class_name[j])
    i=i+1








