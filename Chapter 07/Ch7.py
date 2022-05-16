#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # Read and Write CSV Files

# In[2]:


df = pd.read_csv('books.csv')


# In[3]:


print(df)


# In[8]:


df = pd.read_table('books2.csv', sep=';')


# In[9]:


print(df)


# In[18]:


pd.read_csv('books.csv', nrows=2)


# In[283]:


df.to_csv('books_saved.csv')


# In[5]:


import sys
df.to_csv(sys.stdout)


# In[6]:


df.to_csv(sys.stdout, index=False, header=False)


# In[8]:


df.to_csv(sys.stdout, columns=['Title','Author'],index=False)


# In[15]:


df[1:].to_csv(sys.stdout)


# # Read and Write Microsoft Excel Files

# In[3]:


xlssource = pd.ExcelFile('books.xlsx')
df = pd.read_excel(xlssource, 'Sheet1')
df


# In[5]:


XLSWriter = pd.ExcelWriter('newbooks.xlsx')


# In[6]:


df.to_excel(XLSWriter, 'Sheet1')


# In[7]:


XLSWriter.save()


# # Read data from HTML pages on the web

# In[34]:


url = 'https://en.wikipedia.org/wiki/Rome'
dfs = pd.read_html(url)


# In[35]:


len(dfs)


# In[40]:


dfs = pd.read_html(url, match='Climate data')
len(dfs)


# In[42]:


dfs[0]


# In[44]:


df = dfs[0]


# In[49]:


df = pd.read_csv('books.csv')
df


# In[56]:


df.to_html('books.html')


# # Read and write data from XML files

# In[25]:


import lxml


# In[29]:


df = pd.read_xml('library.xml')
df


# In[30]:


df2 = pd.read_csv('books.csv')
df2.PublicationDate = pd.to_datetime(df2.PublicationDate) 
df2.to_xml('newbooks.xml')


# In[31]:


pd.read_xml('newbooks.xml')


# In[37]:


xml = """<?xml version="1.0" encoding="UTF-8"?>
<library>
  <floor val='1'>
      <book category="crime">
        <title>Death on the Nile</title>
        <author>Agatha Christie</author>
        <year>1937</year>
        <country>United Kingdom</country>
      </book>
      <book category="crime">
        <title>The Hound of the Baskervilles</title>
        <author>Arthur Conan Doyle</author>
        <year>1902</year>
        <country>United Kingdom</country>
      </book>
  </floor>
  <floor val='2'>
      <book category="psichology">
        <title>Psychology and Alchemy</title>
        <author>Carl Gustav Jung</author>
        <year>1944</year>
        <country>Germany</country>
      </book>
      <book category="novel">
        <title>The Castel</title>
        <author>Franz Kafka</author>
        <year>1926</year>
        <country>Germany</country>
      </book>
  </floor>
</library>
"""


# In[34]:


pd.read_xml(xml)


# In[82]:


xsl = """<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
   <xsl:output method="xml" omit-xml-declaration="no" indent="yes"/>
   <xsl:strip-space elements="*"/>
   <xsl:template match="/library">
      <xsl:copy>
        <xsl:apply-templates select="floor"/>
      </xsl:copy>
   </xsl:template>
   <xsl:template match="floor">
      <xsl:apply-templates select="book"/>   
   </xsl:template>
   <xsl:template match="book">
      <xsl:copy>
         <category><xsl:value-of select="@category"/></category>
         <title><xsl:value-of select="title"/></title>
         <author><xsl:value-of select="author"/></author>
         <year><xsl:value-of select="year"/></year>
         <country><xsl:value-of select="country"/></country>
      </xsl:copy>
   </xsl:template>
 </xsl:stylesheet>
 """


# In[83]:


pd.read_xml(xml, stylesheet=xsl)


# # Read and Write data in JSON

# In[285]:


df = pd.read_csv('books.csv')
df


# In[ ]:


df.to_json('books.json')


# In[98]:


import sys
df.to_json(sys.stdout)


# In[97]:


pd.read_json('books.json')


# In[14]:


jsondata = {
"company": "BMW",
"site": "Munchen, Germany",
"production": "Motor Vehicles",
"products": [
{
"product": "Series 1",
"model": " 116d 5-Door Sport Pack"
},
{
"product": "Series 3",
"model": " 330E xDrive Sport Aut"
}
]
}


# In[15]:


pd.read_json(jsondata)


# In[16]:


pd.json_normalize(jsondata)


# In[17]:


pd.json_normalize(jsondata, "products", ["company","site","production"])


# # Write and read data in binary format

# In[110]:


df = pd.read_csv('books.csv')
df


# In[111]:


df.to_pickle('newbooks.dat')


# In[112]:


pd.read_pickle('newbooks.dat')


# # Interact with databases for data exchange

# In[27]:


import sqlite3

query = """
     CREATE TABLE Fruits
     (fruit VARCHAR(20),
      weight REAL(20),
      amount INTEGER,
      price REAL,
      origin VARCHAR(20)
     );"""


# In[28]:


conn = sqlite3.connect('mydata.sqlite')
conn.execute(query)
conn.commit()


# In[29]:


data = [('Apples',12.5,6,100.23,'Germany'),
        ('Cherries',4.5,3,200.50,'Turkey'),
        ('Ananas',19.4,4,300.85,'Madagascar'),
        ('Strawberries',7.8,12,250.33,'Italy'),
       ]

stmt = "INSERT INTO Fruits VALUES (?,?,?,?,?)"
conn.executemany(stmt, data)
conn.commit()


# In[31]:


cursor = conn.execute('SELECT * FROM Fruits')
rows = cursor.fetchall()
rows


# In[32]:


pd.DataFrame(rows, columns=[x[0] for x in cursor.description])


# # Working with Time Data
# 

# In[4]:


pd.Timestamp('2021-11-30')


# In[5]:


pd.Timestamp('2021-Nov-30')


# In[6]:


pd.Timestamp('2021/11/30')


# In[7]:


pd.Timestamp(2021,11,30,0,0,0)


# In[8]:


pd.Timestamp(2021,11,30)


# In[66]:


df = pd.DataFrame({'year': [2019, 2020, 2021],
                   'month': [10, 6, 4],
                   'day': [15, 10, 20]})
print(df)


# In[69]:


ts = pd.to_datetime(df)
print(ts)
print(type(ts))


# In[79]:


t = pd.to_datetime('2021 USA 31--12', format='%Y USA %d--%m', errors='raise')
print(t)


# In[85]:


ts = pd.Series(['2019 USA 31--12','2020 ITA 20--11','2021 USA 10--10'])
print(ts)


# In[89]:


t = pd.to_datetime(ts, format='%Y USA %d--%m', errors='ignore')
print(t)


# In[59]:


pd.Timestamp('2021-Nov-30 11:40', tz='UTC')


# In[48]:


pd.Timestamp('2021-Nov-30 11:40', tz='Europe/Stockholm')


# In[61]:


t = pd.Timestamp('2021-Nov-30 11:40PM')
print(t)
t = t.tz_localize(tz='America/Sao_Paulo')
print(t)


# In[62]:


t2 = t.tz_convert('US/Eastern')
print(t2)


# In[63]:


print(t.year)
print(t.month)
print(t.day)
print(t.hour)
print(t.minute)
print(t.second)
print(t.tz)


# In[64]:


t.hour = 12


# In[90]:


pd.Timedelta('1 Day')


# In[92]:


t = pd.Timestamp('2021-Nov-16 10:30AM')
print(t)
dt = pd.Timedelta('1 Hour')
t = t + dt
print(t)


# In[93]:


t = pd.Timestamp('2021-Nov-16 10:30AM')
t = t + pd.Timedelta('1 Hour')
print(t)


# In[96]:


pd.Timedelta(1, unit="d")


# In[110]:


pd.Timedelta(150045, unit="s")


# In[107]:



Timedelta('3 days 05:34:23')pd.Timedelta('3D 5:34:23')


# In[99]:


t1 = pd.Timestamp('2021-Jan-01 11:30AM')
t2 = pd.Timestamp('2021-Mar-13 4:15PM')
dt = t2 - t1
print(dt)
print(type(dt))


# In[114]:


ts = pd.Series(['00:22:33','00:13:12','00:24:14'])
print(ts)
dt = pd.to_timedelta(ts, errors='raise')
print(dt)


# In[136]:


pd.date_range('2021/01/01', freq='D', periods=10)


# In[139]:


pd.date_range("2021/01/01", "2021/01/10")


# In[153]:


pd.date_range("2021/01/01 8:00","2021/01/01 10:00", freq='5T')


# In[149]:


pd.date_range("2021/01/01 8:00","2021/01/01 10:00", periods=10)


# In[157]:


range = pd.date_range("2021/01/01 8:00","2021/01/01 10:00", periods=10)
tsi = pd.Series(np.random.randn(len(range)), index=range)
print(tsi)


# In[156]:


ts = pd.Series(range)
print(ts)


# In[158]:


tsi["2021/01/01 8:00":"2021/01/01 8:30"]


# In[12]:


df = pd.read_csv('books.csv')
print(df)


# In[13]:


df.PublicationDate


# In[14]:


df['PublicationDate'] = pd.to_datetime(df.PublicationDate)
print(df)


# In[15]:


df.PublicationDate


# In[16]:


df = pd.read_csv('books.csv', parse_dates = [3])
print(df)


# # Working with Text Data

# In[34]:


df = pd.read_csv('books.csv')
print(df)


# In[35]:


print(df.ID)


# In[36]:


print(df.Title)


# In[37]:


print(df.Author)


# In[38]:


df = pd.read_csv('books.csv', dtype='string')
print(df)


# In[39]:


df.PublicationDate


# In[40]:


df.PublicationDate = pd.to_datetime(df.PublicationDate)
df.PublicationDate


# In[41]:


df.Title = df.Title.astype('string')
df.Title


# In[42]:


df.Title.str.upper()


# In[43]:


df.Title.str.lower()


# In[45]:


df['Comment'] = [' Too long for a child   ',
                 'Interesting     book  ',
                 '   Very Impressive']
print(df)


# In[46]:


df.Comment.str.len()


# In[47]:


df.Comment = df.Comment.str.strip()
print(df.Comment)


# In[48]:


print(df.Comment.str.len())


# In[49]:


df.Comment.str.replace('  ','')


# In[52]:


df


# In[53]:


df.Author = df.Author.str.replace(' ',',')
print(df.Author)


# In[54]:


df.Comment.str.replace('book','novel')


# In[55]:


df.Title.str.split()


# In[56]:


df.Author.str.split(',')


# In[57]:


dfw = df.Author.str.split(',', expand=True)
print(dfw)


# In[58]:


type(dfw)


# In[59]:


df[['Author_name','Author_surname']] = df.Author.str.split(',', expand=True)
del df['Author']
df


# In[60]:


df['Author'] = df['Author_name'] + ' ' + df['Author_surname']
df


# In[234]:


df.Author_name.str.cat(df.Author_surname, sep=' ')


# In[61]:


del df['Author_surname']
del df['Author_name']
df


# In[62]:


df['ID1'] = df.ID.str.replace('([A-Z]+)', '', regex=True)
print(df.ID1)


# In[63]:


df['ID2'] = df.ID.str.replace('([0-9]+)', '', regex=True)
print(df.ID2)


# In[64]:


del df['ID']
df


# In[66]:


df['ID'] = df.ID1.str.cat(df.ID2)
df.ID.str.extract('([0-9]+)')


# In[67]:


df.ID.str.extract('([A-Z]+)')


# In[68]:


del df['ID']
df


# In[255]:


df.ID2.str.cat(['USA','ITA','FRA'],sep='-')


# In[271]:


df['temp'] = 'USA'
df.ID2 = df.ID2.str.cat(df['temp'],sep='-')
del df['temp']
print(df.ID2)


# In[272]:


df


# In[273]:


df.ID1.str.isdigit()


# In[274]:


df.ID2.str.isalnum()


# In[275]:


df.Title.str.find('love')


# In[276]:


df.Title.str.find('love') > -1

