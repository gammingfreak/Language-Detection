
# coding: utf-8

# In[1]:


import pandas as pd; import tensorflow; import numpy as np


# In[2]:


n=3
df1 = pd.read_csv("/home/sharmaas/Documents/ML_Project/test/train.csv")
df1.drop(['Unnamed: 0'], axis=1,inplace=True)
print(len(df1))
df1.head()


# In[3]:


np_arr=np.array(df1)


# In[7]:


np_arr2=np.array([[1,"Hi I am mad"],[2,"2 is am mad"]])

interim_list=[]
final_matrix=[None] * len(np_arr2)
for x in range(len(np_arr2)):
    tri_gram_text=""
    for y in range(len(np_arr2[:,1][x])-n+1):
        if (np_arr2[:,1][x][y:y+n] not in interim_list) and (np.char.str_len(np_arr2[:,1][x])<=200):
            tri_gram_text=np_arr2[:,1][x][y:y+n]
            #[x:x+2]
            interim_list.append(tri_gram_text)
    final_matrix[x]=interim_list
    interim_list=[]
np_bag=np.array(final_matrix)
print(np_bag)

