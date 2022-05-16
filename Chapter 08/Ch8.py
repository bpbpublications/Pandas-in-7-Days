#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# # Handling Missing Data

# In[26]:


from numpy import NaN, NAN, nan


# In[5]:


df = pd.read_csv('countries.csv')
df


# In[6]:


df = pd.read_csv('countries.csv', na_values=['Not Known'])
df


# In[8]:


df = pd.read_csv('countries.csv', na_filter=False)
df


# In[10]:


df = pd.read_csv('countries.csv', na_values=['Not Known'], keep_default_na=False)
df


# In[15]:


df = pd.read_csv('books.csv')
df


# In[18]:


df2 = df.reindex(['ID','Author','Title','Price'],axis=1)
df2


# In[17]:


df2['Price'] = [10.00, 20.00, 15.00]
df2


# In[19]:


df.shift(1)


# In[28]:


df1 = pd.read_csv('books_store.csv')
df1


# In[22]:


df.reindex(['ID','Author','Title','Price'],axis=1, fill_value= 0.00)


# In[38]:


df.shift(1, fill_value='Empty')


# In[40]:


df1 = pd.read_csv('books_store.csv')
df1


# In[39]:


pd.merge(df,df1, on = ['ID','Title'], how='right')


# In[ ]:


df = pd.read_csv('countries.csv')
df


# In[41]:


np.count_nonzero(df.isnull())


# In[43]:


np.count_nonzero(df['Area'].isnull())


# In[44]:


df['Area'].value_counts(dropna=False)


# In[45]:


df.dropna()


# In[46]:


df.fillna('Empty')


# In[48]:


df.fillna({'Population':0.0, 'Area': 10.0})


# In[ ]:


df.fillna(0, inplace=True)


# In[49]:


df.fillna(method='ffill')


# In[50]:


df.fillna(method='bfill')


# In[51]:


df.fillna(df.mean())


# In[53]:


df.interpolate()


# In[ ]:


pd.Series([1, 4, None, 5, 12])


# In[ ]:


pd.Series([1, 4, np.nan, 5, 12])


# In[ ]:


pd.Series([1, 4, 5, 12])


# In[ ]:


pd.Series([1, 4, pd.NA, 5, 12])


# In[ ]:


pd.Series([1, 4, pd.NA, 5, None, np.nan], dtype='Int64')


# In[ ]:


pd.array([1, 4, pd.NA, 5, None, np.nan])


# In[ ]:


pd.array([1.0, 4.0, pd.NA, 5.0, None, np.nan])


# In[ ]:


pd.Series([True, False, None, True, False])


# In[ ]:


pd.Series([True, False, np.nan, True, False])


# In[ ]:


pd.Series([True, False, pd.NA, True, False])


# In[ ]:


pd.Series([True, False, pd.NA, True, None, np.nan, False], dtype='boolean')


# In[ ]:


pd.array([True, False, pd.NA, True, None, np.nan, False])


# # Data Replacing

# In[4]:


df = pd.read_csv('clothing.csv')
df


# In[7]:


df.replace('Carmine', 'Red',inplace=True)
df.replace('Bluex','Blue',inplace=True)
df


# # Data Duplicated

# In[8]:


df.duplicated()


# In[9]:


df.drop_duplicates()


# In[14]:


df.drop_duplicates(keep='last', inplace=True)
df


# # Renaming Axis Indexes

# In[76]:


df = pd.read_csv('books.csv')
df


# In[77]:


df = pd.read_csv('books.csv', index_col=0)
df


# In[78]:


df.rename(columns = {'PublicationDate':'Publication'}, inplace=True)
df


# In[93]:


df.rename(index=str.lower, columns=str.upper, inplace=True)
df


# In[96]:


limiter = lambda x: x[:5]
df.index = df.index.map(limiter)
df


# # Sparse DataFrames

# In[29]:


arr = np.random.randn(10000)
arr[arr < 1] = np.nan
df = pd.DataFrame(arr.reshape(100,100))
df


# In[30]:


'{:0.2f} bytes'.format(df.memory_usage().sum() / 1e3)


# In[33]:


sdf = df.astype(pd.SparseDtype("float", np.nan))


# In[34]:


sdf.sparse.density


# In[35]:


'{:0.2f} bytes'.format(sdf.memory_usage().sum() / 1e3)


# # Tidy Data

# In[36]:


df = pd.DataFrame( [[34,22,14],[22,43,22],[14,32,15],[15,22,15]], 
               columns=['Pens','Notebooks','USBSticks'], 
              index=['Sales','HelpDesk','HumanResource','Store'])
df


# In[37]:


df.stack()


# In[38]:


df2 = df.stack().reset_index()
df2


# In[39]:


df2.columns = ['Department','Gadgets','Amount']
df2


# In[23]:


df2 = df.reset_index()
df2 = df2.rename(columns={'index': 'Department'})


# In[24]:


df2.melt(id_vars=['Department'], value_vars=['Pens','Notebooks','USBSticks'],var_name='Gadgets',value_name='Amount')


# In[40]:


df = pd.read_csv('CountrySalaries.csv')
df


# In[13]:


df.melt(id_vars=['Country'], var_name='SalaryRange', value_name='Employees' )


# In[43]:


df = pd.read_csv('Measures.csv')
df


# In[44]:


df1 = df.set_index(['Sensor','Measure'])
df1


# In[45]:


df2 = df1.unstack('Measure')
df2


# In[57]:


df2.reset_index(col_level = -1)


# In[101]:


df = pd.read_csv('PokerHands.csv')
df


# In[102]:


df[['1st_V','1st_S']] = df['1st'].str.split(' ', expand=True)
df


# In[103]:


df[['2nd_V','2nd_S']] = df['2nd'].str.split(' ', expand=True)
df[['3rd_V','3rd_S']] = df['3rd'].str.split(' ', expand=True)
df[['4th_V','4th_S']] = df['4th'].str.split(' ', expand=True)
df[['5th_V','5th_S']] = df['5th'].str.split(' ', expand=True)
del df['1st']
del df['2nd']
del df['3rd']
del df['4th']
del df['5th']
df.index = df['Hand']
del df['Hand']
df


# In[58]:


df = pd.read_csv('CountrySalaries2.csv')
df


# In[21]:


df.melt(id_vars=['Country','Sex'], var_name='SalaryRange', value_name='Employees' )


# In[24]:


df2 = df.pivot_table(index=['Country'],columns='Sex')
df2


# In[104]:


df = pd.read_csv('Aftershaves.csv')
df


# In[65]:


df2 = pd.wide_to_long(df, stubnames=['Component','Vol'], i=['Product'], j='Component_num', sep = '_' )
df2


# In[60]:


df = pd.read_csv('Incomes.csv')
df


# In[81]:


df2 = df.melt(id_vars=['City'],var_name='time_area',value_name='Income')
df2


# In[82]:


df2['year']= df2['time_area'].str[0:4]


# In[83]:


df2['zone']=df2['time_area'].str[-1]


# In[84]:


df2


# In[88]:


df2.drop(['time_area'], axis = 1)


# l
# 
