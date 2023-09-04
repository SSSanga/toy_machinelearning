#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_ROS=pd.read_csv('../datasets/RecurrenceOfSurgery.csv')
df_ROS


# In[3]:


df_ROSL = df_ROS[['Location of herniation','ODI', '입원기간', '통증기간(월)', '수술시간', '수술기법', 'Seg Angle(raw)' ]]
df_ROSL[:2]


# ### 수술기법 TELD와 IELD를 0, 1로 바꾸기

# In[4]:


def tech(str) :
    if str == 'TELD' :
        return 0
    elif str == 'IELD' :
        return 1
    else :
        return None
    


# In[5]:


df_ROSL['수술기법'] = df_ROSL['수술기법'].apply(tech)


# In[6]:


df_ROSL['수술기법']


# In[7]:


df_ROSL.info()


# In[8]:


df_ROSL.isnull().sum()


# ### ODI, 수술기법(int) 제외하고 null 채우기

# In[9]:


mean_value = df_ROSL['통증기간(월)'].mean()


# In[10]:


df_ROSL['통증기간(월)'].fillna(mean_value, inplace=True)


# In[11]:


median_value=df_ROSL['수술시간'].median()


# In[12]:


df_ROSL['수술시간'].fillna(median_value, inplace=True)


# In[13]:


angle_mean = df_ROSL['Seg Angle(raw)'].mean()


# In[14]:


df_ROSL['Seg Angle(raw)'].fillna(angle_mean, inplace=True)


# In[15]:


df_ROSL.isnull().sum()


# ### ODI null값 채우기 (수술기법 제외하고 이용하기)
# - LinearRegression

# In[16]:


df_ROSL_ODI = df_ROSL.drop(columns=['수술기법'])
df_ROSL_ODI[:2]


# In[17]:


df_ROSL_ODI.isnull().sum()


# In[18]:


df_ROSL_ODI_null=df_ROSL_ODI.dropna()
df_ROSL_ODI_null


# In[19]:


target_ODI = df_ROSL_ODI_null['ODI']
labels_ODI = df_ROSL_ODI_null.drop(columns = ['ODI'])


# In[20]:


df_ROSL_ODI_null.columns


# In[21]:


from sklearn.linear_model import LinearRegression 


# In[22]:


model = LinearRegression()


# In[23]:


model.fit(labels_ODI, target_ODI)


# In[24]:


def fill_null_odi_with_model(df, model):
    # null 값이 있는 행을 추출합니다.
    null_rows = df[df['ODI'].isnull()]

    if not null_rows.empty:
        # 'ODI' 열을 예측하기 위한 레이블과 특성을 추출합니다.
        labels = null_rows.drop(columns=['ODI'])
        predicted_odi_values = model.predict(labels)

        # 예측값을 원래 데이터프레임에 적용합니다.
        df.loc[df['ODI'].isnull(), 'ODI'] = predicted_odi_values

    return df


# In[25]:


# 함수를 호출하여 ODI 열의 null 값을 채우세요.
df_filled_ODI = fill_null_odi_with_model(df_ROSL_ODI, model)


# In[26]:


df_filled_ODI


# In[27]:


df_filled_ODI['수술기법']=df_ROSL['수술기법']


# In[28]:


df_filled_ODI[:2] ## ODI null값은 채움


# ### 수술기법 null 채우기
# - LogisticRegression

# In[29]:


df_filled_ODI.isnull().sum()


# In[30]:


df_ROSL_tech = df_filled_ODI.dropna()


# In[31]:


target_tech = df_ROSL_tech['수술기법']
features_tech= df_ROSL_tech.drop(columns=['수술기법'])


# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


model_Logistic = LogisticRegression()


# In[34]:


model_Logistic.fit(features_tech, target_tech)


# In[35]:


def fill_null_tech_with_model(df, model_Logistic):
    # null 값이 있는 행을 추출합니다.
    null_rows = df[df['수술기법'].isnull()]

    if not null_rows.empty:
        # '수술기법' 열을 예측하기 위한 레이블과 특성을 추출합니다.
        features = null_rows.drop(columns=['수술기법'])
        predicted_tech_values = model_Logistic.predict(features)

        # 예측값을 원래 데이터프레임에 적용합니다.
        df.loc[df['수술기법'].isnull(), '수술기법'] = predicted_tech_values

    return df


# In[36]:


# 함수를 호출하여 수술기법 열의 null 값을 채우세요.
df_ROSL_tech = fill_null_tech_with_model(df_filled_ODI, model_Logistic)


# In[37]:


df_ROSL_tech


# In[38]:


df_ROSL_tech.isnull().sum()


# In[39]:


df_ROSL=df_ROSL_tech


# In[40]:


df_ROSL


# In[41]:


df_ROSL.isnull().sum() #final


# ##### Scaling_MinMax
# - OneHotEncoder는 수술기법에 적용하려고 했으나 0,l 두개라서 그냥 if 조건으로 돌려버림
# 
# ###### 스케일링의 이유:
# 
# - 다양한 변수 범위 조정: 다양한 변수의 범위를 조정하여 모든 변수가 동일한 스케일을 가지도록 합니다. 이렇게 하면 모델이 변수 간의 중요도를 쉽게 파악할 수 있습니다.
# 
# - 알고리즘 안정성: 일부 머신러닝 알고리즘은 입력 변수의 스케일에 민감할 수 있습니다. 스케일링을 통해 알고리즘의 안정성을 향상시킵니다.
# 
# - 수렴 속도 향상: 일부 최적화 알고리즘은 데이터가 스케일링되지 않은 경우 더 빠르게 수렴할 수 있습니다.

# In[42]:


target = df_ROSL['Location of herniation']
features = df_ROSL.drop(columns=['Location of herniation'])


# In[43]:


from sklearn.preprocessing import MinMaxScaler


# In[44]:


minMaxScaler=MinMaxScaler()


# In[45]:


features=minMaxScaler.fit_transform(features)
features.shape


# In[ ]:




