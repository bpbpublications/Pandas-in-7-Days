#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # Viewing Data

# In[2]:


dfr = pd.DataFrame(np.random.rand(100).reshape(20,5))
print(dfr)


# In[3]:


dfr


# In[4]:


dfr.head()


# In[5]:


dfr.tail()


# Se si vuole modificare il numero di righe di default da visualizzare, si aggiunge tale valore intero come argomento delle due funzioni.

# In[6]:


dfr.head(3)


# # Selection

# In[119]:


df = pd.DataFrame(np.random.randint(0,9,16).reshape(4,4),
                           index = ['a','b','c','d'],
                           columns = ['A','B','C','D'])
print(df)


# In[120]:


print(df.at['a','A']) #indexing
print(df.iat[0,0])    #integer positioning


# In[121]:


print(df['B'])


# In[122]:


print(df[['B','D']])


# In[123]:


print(df[0])


# In[12]:


print(df['a':'c'])


# In[13]:


print(df[0:1])


# In[14]:


print(df[0:2])


# In[15]:


print(df.loc['a','B'])


# In[16]:


print(df.loc['b':'d','A':'C']) 


# In[17]:


print(df.loc['a':'c', :]) 


# In[18]:


print(df.loc[['a','c'], ['A','D']]) 


# In[19]:


print(df.loc['a':'c', 'A':'B'].join(df.loc['a':'c', 'D']))


# In[20]:


print(df.loc[:,['A','B','D']].loc['a':'c', :])


# In[21]:


print(df.iloc[1,3]) 


# In[22]:


print(df.iloc[0:2,1:3]) 


# In[23]:


print(df.iloc[[1],[0,2]])


# In[24]:


print(type(df.iloc[[1],[0,2]]))


# In[25]:


print(df.iloc[1,[0,2]])


# In[26]:


print(type(df.iloc[1,[0,2]]))


# In[27]:


print(df.at['b','D'])


# In[28]:


print(df.iat[2,1]) 


# In[29]:


print(df.iat[0:2,1:3]) 


# In[30]:


df = pd.DataFrame([[6,0.3,'one', True],[2,5.1,'two', False],[1,4.3,'three', True]], 
                     index=['a','b','c'], 
                     columns=['A','B','C','D'])
print(df)


# In[31]:


df.dtypes


# In[32]:


df2 = df.select_dtypes(include=['bool','int64'])
print(df2)


# In[33]:


df2 = df.select_dtypes(exclude=['bool','int64'])
print(df2)


# # Filtering

# In[34]:


df = pd.DataFrame(np.random.rand(40).reshape(10,4), columns = ['A','B','C','D'])
print(df)


# In[35]:


df['A'] > 0.500000


# In[36]:


df2 = df[ df['A'] > 0.500000 ]
print(df2)


# In[37]:


df2 = df.loc[ df['A'] > 0.500000]
print(df2)


# In[38]:


df.mean() > 0.500000


# In[39]:


df2 = df.loc[:,df.mean() > 0.500000]
print(df2)


# In[40]:


(df['A'] < 0.300000) | (df['A'] > 0.700000)


# In[41]:


df2 = df[(df['A'] < 0.300000) | (df['A'] > 0.700000)]
print(df2)


# In[42]:


cond1 = df['A'] < 0.300000
cond2 = df['A'] > 0.700000
cond3 = df.mean() > 0.500000

df2 = df.loc[cond1 | cond2 ,cond3]
print(df2)


# In[43]:


df = pd.DataFrame(np.random.rand(40).reshape(10,4), columns = ['A','B','C','D'])
print(df)


# In[44]:


df2 = df > 0.1
print(df2)


# In[49]:


(df > 0.1).all()


# In[50]:


(df > 0.1).any()


# In[51]:


(df > 0.1).any().any()


# In[52]:


df = pd.DataFrame(np.arange(9).reshape(3,3), columns = ['A','B','C'])
print(df)


# In[53]:


df2 = df.isin([0,3])
print(df2)


# In[54]:


print(df[df2])


# In[55]:


df2 = df[df.isin([0,3])].dropna(thresh=1).dropna(axis=1)
print(df2)


# # Editing

# In[56]:


df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], 
                     index=['a','b','c'], 
                     columns=['A','B','C'])
print(df)


# In[57]:


df['D'] = 17 
print(df)


# In[58]:


df['D'] 


# In[59]:


df['E'] = [13,14,15]
print(df)


# In[60]:


df['D'] = ['one','two', 'three']
print(df)


# In[61]:


sr = pd.Series([10,11,12],index=['a','b','c'])
df['D'] = sr 
print(df)


# In[62]:


del df['E']
print(df)


# In[63]:


sr = df.pop('D')  
print(type(sr))   
print(df)


# In[64]:


df.insert(1, 'Z', sr) 
print(df)


# In[65]:


df.insert(3, 'B2', df['B'])
print(df)


# In[66]:


df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], 
                     index=['a','b','c'], 
                     columns=['A','B','C'])
print(df)


# In[67]:


print(df['A'])


# In[68]:


print(df.A)


# In[69]:


t = df.loc['a']
print(type(t)) 


# In[70]:


df.loc['d'] = 12 
print(df)


# In[71]:


df.loc['d'] = 10 
print(df)


# In[72]:


df.loc['d'] = [10, 11, 12] 
print(df)


# In[73]:


sr = pd.Series([13,14,15], index=['A','B','C'], name="e") #aggiungi riga con series
df.loc['d'] = sr
print(df)


# In[74]:


del df.loc['d']


# In[75]:


df2 = df.drop(['a'],axis=0) 
print(df2)


# In[76]:


df3 = df.drop(['A'],axis=1) 
print(df3)


# In[77]:


df2 = df.drop(['a','c'],axis=0) 
print(df2)


# In[78]:


df = df.drop(['a'],axis=0) 
print(df)


# In[79]:


df = pd.DataFrame(np.random.randint(1,10,9).reshape(3,3), 
                     index=['a','b','c'], 
                     columns=['A','B','C'])
print(df)


# In[80]:


df2 = df.assign(D = df['A']*2)
print(df2)


# In[81]:


df = df.assign(D = df['A']*2)
print(df)


# In[82]:


df = df.assign(D = np.sqrt(df['A']))
print(df)


# In[83]:


df2 = df.assign(D = np.repeat(1,3))
print(df2)


# # Descriptive Statistics

# In[84]:


df = pd.DataFrame(np.random.rand(40).reshape(10,4))
print(df)


# In[85]:


df.describe() 


# In[86]:


df2 = pd.DataFrame([[1,'one', True],[2,'two',False],
                    [3,'three',True],[4,'four',False],[5,'five', False]], 
                   columns=['numb','word','bool'])
df2.describe()


# In[87]:


df2[['word','bool']].describe()


# In[88]:


df2.describe(include=['int'])


# In[89]:


df.mean(0) #df.mean(axis=0) #df.mean()


# In[90]:


df.mean(1) #df.mean(axis=1)


# In[91]:


df.std()


# In[92]:


std = ( df - df.mean())/ df.std()
print(std.mean())
print(std.std())


# # Trasposition, Sorting and Reindexing

# In[93]:


df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], 
                     index=['a','b','c'], 
                     columns=['A','B','C'])
print(df)


# In[94]:


print(df.T)


# In[95]:


df = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['b','d','c','a'], 
                     columns=['C','D','A','B'])
print(df)


# In[96]:


df2 = df.sort_index()
print(df2)


# In[97]:


df2 = df.sort_index(ascending=False)
print(df2)


# In[98]:


df2 = df.sort_index(axis=1)
print(df2)


# In[99]:


df2 = df.sort_index(axis=1, ascending=False)
print(df2)


# In[100]:


df2 = df.sort_index(axis=0).sort_index(axis=1)
print(df2)


# In[101]:


df2 = df.sort_values(by='A')
print(df2)


# In[102]:


df['A'] = [0,0,2,2]
df2 = df.sort_values(by=['A','B'])
print(df2)


# In[118]:


df2 = df.sort_index(axis=0)
df2.loc['a','A'] = 0
print(df)
print(df2)


# In[104]:


df = df.sort_values(by='A')
print(df)


# # Reindexing

# In[105]:


df = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['b','d','c','a'], 
                     columns=['C','D','A','B'])
print(df)


# In[106]:


df2 = df.reindex(['a','b','c','d'],axis=0)
print(df2)


# In[107]:


df2.iloc[1,1] = 0
print(df2)
print(df)


# In[108]:


df2 = df.reindex(['A','B','C','D'],axis=1)
print(df2)


# In[109]:


df2 = df.reindex(['A','E','C','D',],axis=1)
print(df2)


# In[110]:


dfo = pd.DataFrame(np.zeros(16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
print(dfo)


# In[111]:


df2 = df.reindex_like(dfo)
print(df2)


# In[112]:


dfo = pd.DataFrame(np.zeros(16).reshape(4,4), 
                     index=['a','b','c','e'], 
                     columns=['A','B','W','Z'])
print(dfo)


# In[113]:


df2 = df.reindex_like(dfo)
print(df2)


# In[ ]:





# In[ ]:





# In[ ]:




