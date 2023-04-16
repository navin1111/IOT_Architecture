#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


df=pd.read_csv(r'C:\Users\navin\Downloads\IOT-temp.csv\IOT-temp2.csv')
df.head()


# In[3]:


df.info()


# In[4]:


pd.value_counts(df["room_id/id"].unique())


# In[5]:


df.drop(["room_id/id"], axis = 1, inplace = True) 
df.head()


# In[6]:


df['noted_date']=pd.to_datetime(df['noted_date'])
df_new=df.set_index('noted_date')
df_new.head()


# In[7]:


sum=df.groupby(df['out/in']).count()['id'].iloc[0]+df.groupby(df['out/in']).count()['id'].iloc[1]

print(" sensor'In' is {0} % and sensor'Out' is {1} % of total".format(round(df.groupby(df['out/in']).count()['id'].iloc[0]/sum*100),
    round(df.groupby(df['out/in']).count()['id'].iloc[1]/sum*100)))


# In[8]:


df_new1=df_new.groupby(df_new.index).mean()
df_new2=df_new.groupby([df_new.index,"out/in"]).mean()


# In[9]:


df_new1.plot()


# In[10]:


pd.value_counts(df['noted_date'])


# In[11]:


df_new2.loc['2018-12-09 03:09:00']


# In[12]:


df_new_in=df_new[df_new["out/in"]=="In"].groupby(["noted_date"]).mean()
df_new_out=df_new[df_new["out/in"]=="Out"].groupby(["noted_date"]).mean()
plt.plot(df_new_in.loc[:'2018-9-09 03:09:00'], '--')
plt.plot(df_new_out.loc[:'2018-9-09 03:09:00'], '-')
plt.xlabel('date')
plt.ylabel('temperature')
# plt.xlim(right=3)


# In[13]:


from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from scipy.stats import normaltest
df_new1
df_new1_mm = minmax_scale(df_new1)
scaler = StandardScaler()
df_new1_ss = scaler.fit_transform(df_new1)
print(df_new1_mm,df_new1_ss)


# In[14]:


_, p = normaltest(df_new1_ss.squeeze())
print(f"significance: {p:.2f}")
_, p = normaltest(df_new1_mm.squeeze())
print(f"significance: {p:.2f}")


# In[15]:


plt.scatter(df_new1, df_new1_ss, alpha=0.3)
plt.scatter(df_new1, df_new1_mm, alpha=0.3)

plt.ylabel("standard scaled")
plt.xlabel("original");


# In[16]:


plt.plot(df_new1_ss)
plt.plot(df_new1_mm)
plt.xlabel('time')
plt.ylabel('temproture')


# In[17]:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_new1_ss, model='additive', period=52)
result.plot()


# In[18]:


import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
h=sm.tsa.arima.ARIMA(endog=df_new1_ss,order=(1,1,3))
model=h.fit()
model.summary()


# In[ ]:




