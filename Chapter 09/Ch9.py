#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


import matplotlib.pyplot as plt


# # Matplotlib

# In[4]:


x = np.linspace(0, 10, 50)
y = 2 * np.sin(2 * x) + x


# In[5]:


fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()


# In[6]:


fig, ax = plt.subplots()
ax.plot(x, y, color='red', linewidth=2.0)
ax.set(xlim=(0, 7), xticks=np.arange(1, 7),
       ylim=(0, 8), yticks=np.arange(1, 8))
plt.show()


# In[7]:


fig, ax = plt.subplots()
y2 = x
ax.plot(x, y, color='red', linewidth=2.0)
ax.plot(x, y2, '--', color='blue', linewidth=2.0 )
ax.set(xlim=(0, 7), xticks=np.arange(1, 7),
       ylim=(0, 8), yticks=np.arange(1, 8))
ax.legend(['y=f(x)','y=x'])
plt.show()


# In[8]:


x = np.linspace(0, 10, 50)
y = 2 * np.sin(2 * x) + x
ye = np.random.normal(y,0.5)


# In[9]:


fig, ax = plt.subplots()
ax.scatter(x, ye)
plt.show()


# In[10]:


fig, ax = plt.subplots()
sizes = np.random.uniform(15, 500, len(x))
colors = np.random.uniform(0, 30, len(x))
ax.scatter(x, ye, s=sizes, c=colors)
ax.plot(x, y, ':', color='red', linewidth=2.0)
ax.set(xlim=(2, 8), xticks=np.arange(2, 8),
       ylim=(2, 8), yticks=np.arange(2, 8))
plt.show()


# In[11]:


x = 1 + np.arange(7)
y = 2 * np.sin(2 * x) + x


# In[12]:


fig, ax = plt.subplots()
ax.bar(x, y)
plt.show()


# In[13]:


fig, ax = plt.subplots()
ax.pie(y)
plt.show()


# In[14]:


fig, ax = plt.subplots()
colors = plt.get_cmap('Oranges')(np.linspace(0.2, 0.7, len(x)))
ax.pie(y,radius=1,colors=colors,wedgeprops={"linewidth": 1,"edgecolor": "white"},autopct="%.2f")
ax.legend(x, bbox_to_anchor=(0, 1), loc='best', ncol=1)
plt.show()


# In[15]:


x = 1 + np.arange(7)
y = tuple(2 * np.sin(2 * x) + x)
s = tuple(np.random.uniform(0.5,1.5,7))
D = np.random.normal(y,s,(100, 7))
fig, ax = plt.subplots()
ax.boxplot(D)
plt.show()


# In[110]:


url = 'https://en.wikipedia.org/wiki/Rome'
dfs = pd.read_html(url, match='Climate data')
temperatures = dfs[0]
temperatures


# In[111]:


temperatures = temperatures.droplevel(0, axis=1)
temperatures = temperatures.drop([8])
temperatures = temperatures.set_index('Month')
del temperatures['Year']
temperatures


# In[112]:


temperatures = temperatures.apply(lambda x: x.str.split('(').str.get(0))
temperatures


# In[113]:


temperatures = temperatures.apply(lambda x: x.str.replace(r'[^\x00-\x7F]+','-', regex=True))
temperatures = temperatures.astype(float)
temperatures


# In[114]:


temperatures.index = temperatures.index.str.split('(').str.get(0).str.replace('°C','').str.strip()
temperatures


# In[115]:


dft = temperatures.transpose()
dft


# In[117]:


dft['Record high'].plot()


# In[118]:


dft.iloc[:, 0:5].plot() 


# In[119]:


plt.figure()
dft.iloc[:, 0:5].plot() # Graphic a DataFrame
plt.axhline(0, color="k", linestyle = ':');
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
          ncol=3, fancybox=True, shadow=True)
plt.show()


# In[120]:


dft['Daily mean'].plot(kind='bar')


# In[121]:


dft['Daily mean'].plot.bar()


# In[122]:


dft.iloc[:, 0:4].plot.bar()


# In[123]:


dft.iloc[:, 0:4].plot.bar(figsize=(10,4))


# In[124]:


dft.iloc[:, 0:4].plot.barh(figsize=(4,10))


# In[125]:


plt.figure()
colors = plt.get_cmap('Blues')(np.linspace(0.2,0.7, len(dft['Daily mean'])))
dft['Daily mean'].plot.pie(
             figsize=(5,5),
             autopct="%.2f",
             colors= colors
);
plt.show()


# In[126]:


dft.iloc[:, 5:7].plot.pie(subplots=True, figsize=(8, 4), legend=None)


# In[127]:


dft.plot.scatter(x='Daily mean', y='Average low')


# # Seaborn

# In[128]:


import seaborn as sns


# In[129]:


df = pd.read_csv('MoneySpent.csv')
df.head()


# In[130]:


sns.relplot(data=df,x='Age',y='Amount',hue='Sex')


# In[131]:


sns.relplot(data=df,kind='line', x='Age',y='Amount',style='Sex',hue='Sex')


# In[132]:


sns.lmplot(data=df[df['Age']> 46], x='Age', y='Amount', hue='Sex')


# In[133]:


sns.lmplot(data=df[df['Age']< 46], x='Age',y='Amount',hue='Sex')


# In[134]:


sns.displot(data=df, x='Age', kde=True)


# In[135]:


sns.displot(data=df, x='Age', hue='Sex', kde=True)


# In[136]:


sns.displot(data=df, kind='ecdf', x='Age', hue='Sex', rug=True)


# # Reporting

# In[7]:


df = pd.read_csv('MoneySpent.csv')
df


# In[45]:


df.describe()


# ### Pandas Profiling

# In[46]:


#Importing the function
import pandas_profiling as pdp


# In[47]:


profile = pdp.ProfileReport(df, title='Money spent by day', explorative = True)
profile


# # Report with PDF

# In[5]:


from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('myReport.html')


# In[8]:


plot = df.plot.scatter(x='Age', y='Amount')
fig = plot.get_figure()
filename = 'file:///C:/Users/nelli/Documents/myPandas/graph.png'
fig.savefig('graph.png')


# In[9]:


template_vars = {'title' : 'Spent by Day',
                 'dataframe': df.to_html(),
                 'matplot' : filename
}
html_out = template.render(template_vars)
html_file = open('myReport.html','w') 
html_file.write(html_out)
html_file.close()


# In[11]:


import pdfkit as pdf
path_wkthmltopdf =  'C:\Programmi\wkhtmltopdf\\bin\wkhtmltopdf.exe'
config = pdf.configuration(wkhtmltopdf=path_wkthmltopdf)
options = {
            "enable-local-file-access": None
}
pdf.from_file('myReport.html', 'myReport.pdf', configuration=config, options = options)


# In[ ]:





# In[ ]:




