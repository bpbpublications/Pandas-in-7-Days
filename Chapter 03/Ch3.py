#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # SERIES

# In[2]:


lst = [ 5 ,9 ,11, -1, 4 ]
nar = np.array([ 5 ,9 ,11, -1, 4 ])


# In[3]:


print(lst)
print(nar)


# In[4]:


ser1 = pd.Series([5, 9, 11, -1, 4])


# In[5]:


print(ser1)


# In[6]:


print(ser1.values)


# In[7]:


print(ser1.index)


# In[8]:


print(ser1[2])


# In[9]:


print(nar[2])


# In[10]:


ser2 = pd.Series([5, 9, 11, -1, 4], 
                  index = ['a','b','c','d','a'])


# In[11]:


print(ser2)


# In[12]:


print(ser2['a'])


# In[13]:


print(ser1)


# In[14]:


ser1.index = ['a','b','c','d','a']


# In[15]:


print(ser1)


# In[16]:


idx = pd.Index(['a','b','c','d','a'])
ser3 = pd.Series( [5, 9, 11, -1, 4], index = idx)


# In[17]:


print(ser3)


# In[18]:


ser1.name = "First series"
print(ser1)


# In[19]:


ser1.index.name = "Characters"
print(ser1)


# In[20]:


ser4 = pd.Series([5, 9, 11, -1, 4], 
                  index = ['a','b','c','d','a'], dtype=float)
print(ser4)


# In[21]:


nda = np.array([7,5,4,1,-11])
print(nda)
print(nda.dtype)


# In[22]:


ser5 = pd.Series(nda)
print(ser5)


# In[23]:


ser5 = pd.Series(nda, dtype=np.int64)
print(ser5)


# In[24]:


ser6 = pd.Series(np.random.randn(5))
print(ser6)


# In[25]:


ser7 = pd.Series(np.arange(1,6))
print(ser7)


# In[26]:


ser8 = pd.Series(np.zeros(4))
print(ser8)


# In[27]:


ser9 = pd.Series(4)
print(ser9)


# In[28]:


ser9 = pd.Series(4, index= ['a','b','c','d'])
print(ser9)


# In[29]:


d = { 'a': 12, 'b': -1, 'c': 7, 'd': 3 }
ser10 = pd.Series(d)
print(ser10)


# In[30]:


idx = pd.Index(['a','b','c'])
d = { 'a': 12, 'b': -1, 'c': 7, 'd': 3 }
ser12 = pd.Series(d, idx)
print(ser12)


# In[31]:


idx = pd.Index(['a','b','e'])
d = { 'a': 12, 'b': -1, 'c': 7, 'd': 3 }
ser12 = pd.Series(d, idx)
print(ser12)


# In[32]:


t = [('a',5),('b',2),('c',-3)] #list of tuples
idx, values = zip(*t)


# In[33]:


ser11 = pd.Series(values, idx)
print(ser11)


# In[34]:


t = [['a',2],['b',4],['c',3]]
idx, values = zip(*t)
ser12 = pd.Series(values, idx)
print(ser12)


# # DATAFRAME

# In[35]:


df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6] ,[7, 8, 9]])
print(df1)


# In[36]:


df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6] ,[7, 8, 9]],
                   index = ['a','b','c'],
                   columns = ['A','B','C'])
print(df1)


# In[37]:


df1.index


# In[38]:


df1.columns


# In[39]:


v = df1.values
print(v)
type(v)


# In[40]:


df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6] ,[7, 8, 9]],
                   index = ['a','b','c'],
                   columns = ['A','B','C'],
                   dtype= np.float64) 
print(df1)


# In[41]:


nda = np.array([[1, 2, 3], [4, 5, 6] ,[7, 8, 9]])
df2 = pd.DataFrame(nda, index = ['a','b','c'], columns = ['A','B','C'])
print(df2)


# In[42]:


nda[1] = [4,0,6]
print(nda)


# In[43]:


print(df2)


# In[44]:


nda = np.array([[1, 2, 3], [4, 5, 6] ,[7, 8, 9]])
df2 = pd.DataFrame(nda, index = ['a','b','c'], columns = ['A','B','C'], copy=True)
print(df2)


# In[45]:


nda[1] = [4,0,6]
print(nda)


# In[46]:


print(df2)


# In[47]:


print(df2.iloc[0]) #select first row
print(df2.iloc[:,1]) #select second column


# In[48]:


print(df2.loc['a','B'])


# In[49]:


print(df2.loc['a']) #select first row
print(df2.loc[:,'B']) #select second column


# In[50]:


idx = np.array(['a','b','c'])
col = np.array(['A','B','C'])
nda = np.array([[1, 2, 3], [4, 5, 6] ,[7, 8, 9]])
df3 = pd.DataFrame(nda, index=idx, columns=col)
print(df3)


# In[51]:


df3 = pd.DataFrame(np.arange(1,10).reshape(3,3), index=idx, columns=col)
print(df3)


# In[52]:


df4 = pd.DataFrame(np.zeros(9).reshape(3,3), index=idx, columns=col)
print(df4)


# In[53]:


df5 = pd.DataFrame(np.random.randn(9).reshape(3,3), index=idx, columns=col)
print(df5)


# In[54]:


ser1 = pd.Series([1,4,7], index=['a','b','c'])
ser2 = pd.Series([2,5,8], index=['a','b','c'])
ser3 = pd.Series([3,6,9], index=['a','b','c'])
d = { "A" : ser1,
      "B" : ser2, 
      "C" : ser3}


# In[55]:


df6 = pd.DataFrame(d)
print(df6)


# In[56]:


ser1 = pd.Series([1,4,7,-3,-1], index=['a','b','c','e','f'])
ser2 = pd.Series([2,5,8], index=['d','b','c'])
ser3 = pd.Series([3,6,9,0], index=['a','b','c','d'])
d = { "A" : ser1,
      "B" : ser2, 
      "C" : ser3}
df7 = pd.DataFrame(d)
print(df7)


# In[57]:


df7 = pd.DataFrame(d, index=['a','b','c'])
print(df7)


# In[58]:


nda1 = np.array([1,4,7]) 
nda2 = np.array([2,5,8])
nda3 = np.array([3,6,9])
idx = np.array(['a','b','c'])
d = { "A" : nda1,
      "B" : nda2, 
      "C" : nda3}
df8 = pd.DataFrame(d, idx)
print(df8)


# In[59]:


str_array = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                   dtype=[('A', '<f8'), ('B', '<i8'),('C','i4')])
print(str_array)


# In[60]:


print(str_array['A'])
print(str_array[1])


# In[61]:


df9 = pd.DataFrame(str_array)
print(df9)


# In[62]:


df9 = pd.DataFrame(str_array, index=['a','b','c'])
print(df9)


# In[63]:


rec_array = np.rec.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                        dtype=[('A', '<f8'), ('B', '<i8'),('C','i4')])
print(rec_array)


# In[64]:


print(rec_array.A)


# In[65]:


df10 = pd.DataFrame(rec_array, index=['a','b','c'])
print(df10)


# In[66]:


list_dicts = [{ 'A': 1, 'B': 2, 'C': 3},
              { 'A': 4, 'B': 5, 'C': 6},
              { 'A': 7, 'B': 8, 'C': 9},]
print(list_dicts)


# In[67]:


df11 = pd.DataFrame(list_dicts, index=['a','b','c'])
print(df11)


# In[68]:


from dataclasses import make_dataclass


# In[69]:


Dataset = make_dataclass("Dataset", [('A', int),('B', int),('C', int)])
df12 = pd.DataFrame([Dataset(1,2,3), Dataset(4,5,6), Dataset(7,8,9)], index=['a','b','c'])
print(df12)


# In[70]:


from collections import namedtuple


# In[71]:


Dataset = namedtuple('Dataset', 'A B C')
df13 = pd.DataFrame([Dataset(1,2,3), Dataset(4,5,6), Dataset(7,8,9)], index=['a','b','c'])
print(df13)


#  # MultiIndex DataFrame

# In[72]:


dict_tuples = { ('X', 'A'): {('x','a'): 1, ('x','b'): 4, ('x','c'): 7, ('y','d'): 0, ('y','e'): 0 },
                ('X', 'B'): {('x','a'): 2, ('x','b'): 5, ('x','c'): 8, ('y','d'): 0, ('y','e'): 0 },
                ('X', 'C'): {('x','a'): 3, ('x','b'): 6, ('x','c'): 9, ('y','d'): 0, ('y','e'): 0 },
                ('Y', 'D'): {('x','a'): -1, ('x','b'): -3, ('x','c'): -5},
                ('Y', 'E'): {('x','a'): -2, ('x','b'): -4, ('x','c'): -6}
              }


# In[73]:


multidf = pd.DataFrame(dict_tuples)
print(multidf)


# In[74]:


print(multidf['X'].loc['x'])


# In[75]:


print(multidf['X']['A'])


# In[76]:


print(multidf.columns)
print(multidf.index)


# In[77]:


print(multidf.columns.levels)
print(multidf.index.levels)


# In[78]:


dfn = print(multidf['X'].loc['x'])
print(dfn)


# In[ ]:




