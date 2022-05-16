#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# # Adding data to a DataFrame
# 

# In[3]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
print(df1)


# In[4]:


df2 = pd.DataFrame(np.zeros(16).reshape(4,4), 
                     index=['e','f','g','h'], 
                     columns=['A','B','C','D'])
print(df2)


# ## append()

# In[5]:


df = df1.append(df2)
print(df)


# In[62]:


ser = pd.Series([7,7,7,7], index=['A','B','C','D'])
print(ser)


# In[29]:


df1.append(ser)


# In[63]:


df2 = df1.append(ser,ignore_index=True)
print(df2)


# In[64]:


ser = pd.Series([7,7,7,7], index=['A','B','C','D'], name='e')
print(ser)


# In[65]:


df2 = df1.append(ser)   # df1.append(ser,ignore_index=False)
print(df2)


# In[66]:


ser.name = 'z'
df2 = df1.append(ser)
print(df2)


# In[67]:


df2 = df1.append([ser,ser+1,ser+2])
print(df2)


# ## concat()

# In[68]:


df = pd.concat([df1,df2])
print(df)


# In[69]:


df = pd.concat([df1,df2,df1])
print(df)


# In[70]:


ser = pd.Series([7,7,7,7], index=['a','b','c','d'], name='E')
print(ser)


# In[71]:


df = pd.concat([df1,ser],axis=1)
print(df)


# In[72]:


ser = pd.Series([7,7,7,7,7], index=['a','c','d','e','f'], name='E')
df = pd.concat([df1,ser],axis=1)
print(df)


# In[73]:


ser = pd.Series([7,7,7,7,7], index=['a','c','d','e','f'], name='E')
df = pd.concat([df1,ser],axis=1,join='inner')
print(df)


# In[74]:


ser = pd.Series([7,7,7,7], index=['a','b','c','d'], name='E')
ser2 = ser + 1
ser3 = ser + 2
df = pd.concat([df1,ser,ser2,ser3,df1],axis=1)
print(df)


# ## Merge
# 

# In[75]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
df2 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['b','c','d','e'], 
                     columns=['B','C','D','E'])
print(df1)
print(df2)


# In[76]:


df = pd.merge(df1,df2, on = 'B', how='left')
print(df)


# Utilizzando invece una logica RIGHT sul joining si ottiene il risultato perfettamente opposto a quello precedente.

# In[77]:


df = pd.merge(df1,df2, on = 'B', how='right')
print(df)


# In[78]:


df = pd.merge(df1,df2, on = 'B', how='outer')
print(df)


# In[79]:


df = pd.merge(df1,df2, on = 'B', how='inner')
print(df)


# In[80]:


df = pd.merge(df1,df2, on = 'B')
print(df)


# # Arithmetic with DataFrames

# In[81]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
print(df1)


# In[82]:


df2 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
print(df2)


# In[83]:


print(df1 + df2)


# In[84]:


print(df1 - 2*df2)


# In[85]:


print(df1 > df2)


# In[87]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
df2 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['B','C','D','E'])
print(df1)
print(df2)


# In[88]:


print(df1 + df2)


# In[91]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
df2 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['b','c','d','e'], 
                     columns=['A','B','C','D'])
print(df1)
print(df2)


# In[92]:


print(df1 + df2)


# In[93]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
ser = pd.Series(np.random.randint(10, size=4), 
                     index=['A','B','C','D'])
print(df1)
print(ser)


# In[94]:


print(df1 + ser)


# In[95]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
ser = pd.Series(np.random.randint(10, size=4), 
                     index=['B','C','D','E'])
print(df1)
print(ser)


# In[96]:


print(df1 + ser)


# # Flexible binary arithmetic methods

# In[98]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
df2 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['b','c','d','e'], 
                     columns=['A','B','C','D'])
print(df1)
print(df2)


# In[99]:


df = df1.add(df2)
print(df)


# In[100]:


df = df1 + df2  
print(df)


# In[45]:


ser = pd.Series([1,1,1,1],index=['A','B','C','D'])
print(ser)


# In[101]:


df = df1.add(ser)
print(df)


# In[102]:


df = df1 + ser
print(df)


# In[103]:


ser = pd.Series([1,2,3,4], index=['a','b','c','d'])

df = df1.add(ser, axis=0) # df.add(df2, axis='index')
print(df)


# In[106]:


df = df1 + ser
print(df)


# In[107]:


tuples = [('x', 'a'), ('x', 'b'), ('y', 'a'), ('y', 'b')]
index = pd.MultiIndex.from_tuples(tuples, names=['first','second'])
tuples = [('X', 'A'), ('X', 'B'), ('Y', 'A'), ('Y', 'B')]
columns = pd.MultiIndex.from_tuples(tuples, names=['high','low'])
dfm = pd.DataFrame(np.arange(1,17).reshape(4,4) , index = index, columns = columns)
print(dfm)


# In[108]:


ser = pd.Series([10,20], index=['A','B'])
df = dfm.sub(ser, level='low')
print(df)


# In[109]:


ser = pd.Series([10,20], index=['X','Y'])
df = dfm.sub(ser, level='high')
print(df)


# In[110]:


ser = pd.Series([10,20], index=['a','b'])
df = dfm.sub(ser, level='second',axis=0)
print(df)


# In[111]:


ser = pd.Series([10,20], index=['x','y'])
df = dfm.sub(ser, level='first',axis=0)
print(df)


# In[112]:


df3 = pd.DataFrame(np.random.randint(10, size=4).reshape(2,2), 
                     index=['a','b'], 
                     columns=['A','B'])
print(df3)


# In[113]:


df= dfm.add(df3, axis=1, level=1)
print(df)


# In[115]:


row = df1.iloc[0]
print(row)
column = df1['A']
print(column)


# In[116]:


df = df1.mul(column, axis=0)
print(df)


# In[117]:


df = df1.mul(row, axis=1)
print(df)


# In[118]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
df2 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['b','c','d','e'], 
                     columns=['A','B','C','D'])
print(df1)
print(df2)

column = df2['A']
print(column)


# In[119]:


df = df1.loc['b':'d',['A','B']]
print(df)


# In[120]:


df = df2.loc['b':'d',['A','B']].mul(column, axis=0)
print(df)


# ## The relative binary methods

# In[121]:


# Operazioni binarie relative

print(df / 1)
print(df.div(1))


# In[122]:


print(1 / df)
print(df.rdiv(1))


# ## Boolean operations

# In[123]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
df2 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','C','D'])
print(df1)
print(df2)


# In[124]:


print(df1 > df2)


# In[125]:


df = df1[df1 > df2]
print(df)


# In[126]:


df = df1.lt(df2) # df1 > df2 
print(df)


# In[127]:


df = df1.lt(df2['A'], axis=0)
print(df)


# # Aligning 

# In[6]:


df1 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['a','b','c','d'], 
                     columns=['A','B','D','E'])
df2 = pd.DataFrame(np.random.randint(10, size=16).reshape(4,4), 
                     index=['b','c','d','e'], 
                     columns=['A','C','D','E'])
print(df1)
print(df2)


# In[130]:


df1n, df2n = df1.align(df2)
print(df1n)
print(df2n)


# In[131]:


df1n, df2n = df1.align(df2, join='inner')
print(df1n)
print(df2n)


# In[132]:


df1n , df2n = df1.align(df2, join='left')
print(df1n)
print(df2n)


# In[134]:


df1n , df2n = df1.align(df2, join='right')
print(df1n)
print(df2n)


# In[ ]:




