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


# In[33]:


df_ROSL.info()


# In[6]:


df_ROSL.isnull().sum()


# In[7]:


df_ROSL_dropna = df_ROSL.dropna()
df_ROSL_dropna.isnull().sum()
df_ROSL_dropna.info()


# #### learning
# - raw data를 target_train_split를 나눠서 모델 학습 및 GridSearchCV를 하고 평가를 한다. 
# - 설명변수의 범주형: 수술기법을 제외하고 모델학습시킨다. 

# In[8]:


df_ROSL_dropna.columns


# In[9]:


target = df_ROSL_dropna['Location of herniation']
features = df_ROSL_dropna[['ODI','입원기간','통증기간(월)','수술시간','Seg Angle(raw)']]
target.shape, features.shape


# In[10]:


from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=111)
features_train.shape, features_test.shape, target_train.shape, target_test.shape


# In[11]:


from sklearn.tree import DecisionTreeClassifier


# In[12]:


model = DecisionTreeClassifier()


# In[13]:


from sklearn.model_selection import GridSearchCV


# In[14]:


hyper_params = {'min_samples_leaf' : [2, 9], 
                'max_depth' : [2, 9], 
                'min_samples_split' : [2, 9]}


# #### 평가 score : 분류-정확도, 예측-R squre

# In[15]:


from sklearn.metrics import f1_score


# In[20]:


grid_search = GridSearchCV(model, param_grid = hyper_params, cv=3, verbose=1)


# In[21]:


grid_search.fit(features_train, target_train)


# In[22]:


grid_search.best_estimator_


# In[23]:


grid_search.best_score_, grid_search.best_params_


# In[24]:


best_model = grid_search.best_estimator_
best_model


# In[25]:


target_test_predict = best_model.predict(features_test)
target_test_predict


# In[26]:


from sklearn.metrics import classification_report


# In[27]:


print(classification_report(target_test, target_test_predict))


# In[29]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# In[31]:


confusion_matrix(target_test, target_test_predict)


# ### 전처리랑 학습 파일 합쳐서 서비스 작성하기

# In[ ]:




