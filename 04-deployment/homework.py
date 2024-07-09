#!/usr/bin/env python
# coding: utf-8

# ## Q1. Notebook
# We'll start with the same notebook we ended up with in homework 1. We cleaned it a little bit and kept only the scoring part. You can find the initial notebook here.
# 
# Run this notebook for the March 2023 data.
# 
# What's the standard deviation of the predicted duration for this dataset?
# 
# - 1.24
# - 6.24
# - 12.28
# - 18.28
# 
# Answer: 6.24

# In[1]:


# get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


# get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd
import sys
import numpy

print(numpy.version.version)

# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[5]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[6]:

year = int(sys.argv[1])
month = int(sys.argv[2])


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[7]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[8]:


y_pred.std()

print(y_pred.mean())


# ## Q2. Preparing the output
# Like in the course videos, we want to prepare the dataframe with the output.
# 
# First, let's create an artificial `ride_id` column:
# 
# ``` python
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
# ```
# 
# Next, write the ride id and the predictions to a dataframe with results.
# 
# Save it as parquet:
# 
# ```python
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
# ```
# What's the size of the output file?
# 
# - 36M
# - 46M
# - 56M
# - 66M
# Note: Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the dtypes of the columns and use `pyarrow`, not `fastparquet`.
# 
# Answer: 66M

# In[12]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = df[['ride_id']].copy()
df_result['pred_duration'] = y_pred

df_result


# In[15]:


df_result.to_parquet(
    "output.parquet",
    engine='pyarrow',
    compression=None,
    index=False
)


# In[16]:


# get_ipython().system('du -h output.parquet ')


# In[ ]:




