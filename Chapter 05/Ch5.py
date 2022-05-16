#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # Shifting

# In[7]:


df = pd.DataFrame(np.arange(1,10).reshape(3,3),
                  index=['a','b','c'],
                  columns=['A','B','C'])
print(df)


# In[8]:


df2 = df.shift(1)
print(df2)


# In[9]:


df2 = df.shift(1, fill_value=0)
print(df2)


# In[11]:


df2 = df.shift(1)
df2.iloc[0] = df.iloc[2]
print(df2)


# In[13]:


df2 = df.shift(1)
df2 = df2.drop(['a'])
print(df2)


# In[14]:


df2 = df.shift(-2)
print(df2)


# In[15]:


df2 = df.shift(1, axis=1)
print(df2)


# In[16]:


df2 = df.shift(-1, axis=1)
print(df2)


# # Reshape

# In[2]:


df = pd.DataFrame(np.arange(1,10).reshape(3,3),
                index=['a','b','c'],
                columns=['A','B','C'])
print(df)


# In[3]:


dfs = df.stack()
print(dfs)


# In[4]:


print(dfs.unstack())


# In[6]:


index = pd.MultiIndex.from_tuples([('x','a'),
                                   ('x','b'),
                                   ('y','a'),
                                   ('y','b')])
columns = pd.MultiIndex.from_tuples([('X','A'),
                                     ('X','B'),
                                     ('Y','A'),
                                     ('Y','B')])
dfm = pd.DataFrame(np.arange(1,17).reshape(4,4),
                     index=index,
                     columns=columns)
print(dfm)


# In[7]:


print(dfm.stack(0)) # print(dfm.stack(level=0))


# In[8]:


print(dfm.unstack(0)) #print(dfm.stack(level=0))


# ## Pivoting

# In[3]:


df = pd.DataFrame([['red','x',1,1],
                   ['red','x',0,1],
                   ['black','y',0,2],
                   ['red','y',1,0],
                   ['red','y',2,0],
                   ['black','x',2,1]],
                  columns=['A','B','C','D'])
print(df)


# In[4]:


table = pd.pivot_table(df, values=['C','D'],
                       index=['A'],
                       columns=['B'],
                       aggfunc=np.sum)
print(table)


# In[5]:


table = pd.pivot_table(df, values=['C','D'],
                       index=['A'],
                       columns=['B'],
                       aggfunc={'C': np.sum,
                                'D': [min, max, np.mean]})
print(table)


# # Iteration

# In[7]:


ser = pd.Series(np.arange(1,12))
print(ser)


# In[10]:


for i in ser:
    print(i, ' ', np.sqrt(i))


# In[12]:


df = pd.DataFrame(np.arange(1,10).reshape(3,3),
                 index=['a','b','c'],
                 columns=['A','B','C'])
print(df)


# In[16]:


for i in df:
    print(i)


# In[15]:


for i in df:
    print(df[i])


# In[17]:


for i in df:
    for j in df[i]:
        print(j)


# In[18]:


df = pd.DataFrame(np.arange(1,10).reshape(3,3),
                 index=['a','b','c'],
                 columns=['A','B','C'])
print(df)


# In[19]:


for label, ser in df.items():
    print(label)
    print(ser)


# In[20]:


for index, row in df.iterrows():
    print(index)
    print(row)


# In[28]:


for row in df.itertuples():
    print(row)
    print(row.A)
    print(row.B)
    print(row.C)  
    print(row.Index)


# # Apply Functions on a Dataframe

# In[14]:


df = pd.DataFrame(np.random.rand(16).reshape(4,4),
                  index=['a','b','c','d'],
                  columns=['A','B','C','D'])
print(df)


# In[15]:


df2 = np.exp(df)
print(df2)


# In[13]:


type(df2)


# In[16]:


np.asarray(df2)


# In[18]:


np.max(df)


# In[19]:


df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], 
                     index=['a','b','c'], 
                     columns=['A','B','C'])
print(df)


# In[5]:


df.apply(np.sum)


# In[12]:


df.apply(np.sum, axis=1)


# In[20]:


df2 = df.apply(np.sqrt)
print(df2)


# In[21]:


def dubble_up(x):
    return x * 2

df2 = df.apply(dubble_up)
print(df2)


# In[30]:


df.apply(lambda x: x.max() - x.min())


# In[31]:


df.apply(lambda x: x.max() - x.min(), axis=1)


# In[22]:


df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], 
                     index=['a','b','c'], 
                     columns=['A','B','C'])
print(df)


# In[35]:


def maximus(x):
    return np.max(np.max(x))


# In[36]:


df.pipe(maximus)


# In[39]:


df.apply(maximus)


# In[43]:


df.pipe(np.max).pipe(np.max)


# In[44]:


step1 = df.pipe(np.max)
print(step1)
step2 = step1.pipe(np.max)
print(step2)


# In[45]:


def dubble_up(x):
    return x * 2

df.pipe(dubble_up).pipe(np.mean).pipe(lambda x: x-2)


# In[46]:


df.apply(dubble_up).apply(np.mean).apply(lambda x: x-2)


# In[24]:


def adder(a, b):
    return a + b


# In[25]:


df2 = df.pipe(adder, 2)
print(df2)


# In[ ]:


print(df)
df2 = pd.DataFrame(np.repeat(1,9).reshape(3,3), 
                      index=['a','b','c'], 
                      columns=['A','B','C'])
print(df2)
df3 = df.pipe(adder, df2)
print(df3)


# In[28]:


df2 = df.applymap(dubble_up)
print(df2)


# In[7]:


ser = pd.Series([1,3,5,7,9,11])
ser.map(dubble_up)


# # Transform

# In[30]:


df = pd.DataFrame(np.arange(1,10).reshape(3,3),
                 index=['a','b','c'],
                 columns=['A','B','C'])
print(df)


# In[71]:


def double(x):
    return x*2

df2 = df.transform(double)  
print(df2)


# In[72]:


df2 = df.apply(double)
print(df2)


# In[84]:


multidf = df.transform([np.sqrt, double])
print(multidf)


# In[113]:


idx = pd.IndexSlice
dfn = multidf.loc[idx[:], idx[:, 'double']]

dfn.columns = dfn.columns.droplevel(1)
print(dfn)


# In[29]:


df2 = df.transform({
    'A': np.sqrt,
    'B': np.double,
})
print(df2)


# In[77]:


df.transform(np.sum)


# In[78]:


df.apply(sum)


# In[80]:


def adding(x):
    return x[0] + x[1]

df.apply(adding, axis=1)


# In[81]:


# Getting error when trying the same with transform
df.transform(subtract_two, axis=1)


# # Aggregation

# In[148]:


df = pd.DataFrame(np.arange(1,10).reshape(3,3),
                 index=['a','b','c'],
                 columns=['A','B','C'])
print(df)


# In[149]:


df.agg(np.sum)


# In[150]:


multidf = df.agg([np.sqrt, double])
print(multidf)


# In[30]:


df2 = df.agg({
    'A': np.sqrt,
    'B': np.double,
})
print(df2)


# In[153]:


df.agg(np.sum)


# In[154]:


df = df.agg([np.mean, np.sum])
print(multidf)


# In[31]:


df2 = df.agg({
    'A': np.mean,
    'B': np.sum,
})
print(df2)


# In[155]:


df = df.agg([np.mean, np.sum, np.sqrt])
print(multidf)


# # Grouping

# In[33]:


df = pd.DataFrame(np.random.randint(1,5,18).reshape(6,3),
                 index=['a','b','c','a','b','c'],
                 columns=['A','B','C'])
print(df.sort_values(by=['A']))


# In[3]:


df.groupby(by='A')


# In[34]:


for name, group in df.groupby(by='A'):
    print(name)
    print(group)


# In[35]:


df2 = df.groupby(by='A').count()
print(df2)


# In[36]:


df2 = df.groupby(by='A').sum()
print(df2)


# In[37]:


dfg = df.groupby(by='A').max()
print(dfg)


# In[41]:


tuples = [('x', 'a'), ('x', 'b'), ('y', 'a'), ('y', 'b')]
index = pd.MultiIndex.from_tuples(tuples, names=['first','second'])
tuples = [('X', 'A'), ('X', 'B'), ('Y', 'A'), ('Y', 'B')]
columns = pd.MultiIndex.from_tuples(tuples, names=['high','low'])
dfm = pd.DataFrame(np.arange(1,17).reshape(4,4) , index = index, columns = columns)
print(dfm)


# In[42]:


df2 = dfm.groupby(level=0).max()
print(df2)


# In[45]:


df2 = dfm.groupby(level=1).max()
print(df2)


# In[47]:


df2 = dfm.groupby(level=0, axis=1).max()
print(df2)


# In[46]:


df2 = dfm.groupby(level=1, axis=1).max()
print(df2)


# In[38]:


gdf = df.groupby('A').transform(np.sum)
print(gdf)


# In[39]:


gdf = df.groupby('A').apply(np.sum)
print(gdf)


# In[40]:


gdf = df.groupby('A').agg(np.sum)
print(gdf)


# # Categorization

# In[17]:


df = pd.DataFrame(np.arange(1,10).reshape(3,3), 
                       index=['a','b','c'],
                       columns=['A','B','C'])
print(df)


# In[18]:


cat = pd.Categorical(['IV','I','IV'],
                     categories=['I','II','III','IV','V'],
                     ordered=True)
df['cat'] = cat
print(df)


# In[19]:


print(df['cat'].cat.as_ordered())


# In[20]:


print(df.sort_values(by='cat'))


# In[21]:


print(df.groupby('cat').size())


# In[22]:


df['cat'] = ['IV','I','III']
print(df)


# In[23]:


df['cat'] = df['cat'].astype('category')
df['cat'] = df['cat'].cat.set_categories(['I','II','III','IV','V'], ordered=True)
print(df)


# In[24]:


print(df.sort_values(by='cat'))

