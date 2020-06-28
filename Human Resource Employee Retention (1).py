#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[6]:


hr_dt=pd.read_csv(r'D:\data science and deepl learning 20 case studies\datascienceforbusiness-master\hr_data.csv')


# In[7]:


#View the Top 5 rows
hr_dt.head()


# In[8]:


#view the bottom 5 rows
hr_dt.tail()


# In[9]:


hr_dt.describe()


# In[11]:


hr_dt.isnull().sum()


# In[15]:


# To view which columns are categorical,Numerical
hr_dt.info()


# In[16]:


# To check the number of Rows and columns
hr_dt.shape


# In[20]:


# To display uniques in Categorical columns
print(hr_dt.department.unique())
print(hr_dt.salary.unique())


# In[21]:


# To display values in categorical
print(hr_dt.department.value_counts())
print(hr_dt.salary.value_counts())


# # Loading our Evaluation and Emoloyee Satisfaction Data

# In[22]:


satis_dt=pd.read_excel('D:\data science and deepl learning 20 case studies\datascienceforbusiness-master\employee_satisfaction_evaluation.xlsx')


# In[23]:


satis_dt.head()


# Merge the two data frames or Join the tables

# In[24]:


main_dt=hr_dt.set_index('employee_id').join(satis_dt.set_index('EMPLOYEE #'))


# In[26]:


main_dt=main_dt.reset_index()


# In[27]:


main_dt.head()


# Lets check for the missing values?

# In[28]:


main_dt.isnull().sum()


# # Lets fill the missing values impute with the Mean

# In[30]:


main_dt.mean()


# In[32]:


main_dt.fillna(main_dt.mean(),inplace=True)


# In[35]:


main_dt.head()


# In[37]:


#Lets drop the Employee id column 
main_dt_final=main_dt.drop(columns='employee_id')
main_dt_final.head()


# In[38]:


main_dt_final.groupby('department').sum()


# In[40]:


main_dt_final.groupby('department').mean()


# In[41]:


# lets check the class of Left column
main_dt_final.left.value_counts()


# # Displaying your correlation matrix
# 

# In[42]:


import matplotlib.pyplot as plt


# In[44]:


def plot_corr(df,size=10):
    corr=df.corr()
    fig, ax=plt.subplots(figsize=(size,size))
    ax.legend()
    cax=ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)),corr.columns,rotation='vertical')
    plt.yticks(range(len(corr.columns)),corr.columns)
    
plot_corr(main_dt_final)


# # Preparing our Dataset for ML

# In[45]:


#perform one hot encoding on Categorical Data

categorical=['department','salary']
main_dt_final=pd.get_dummies(main_dt_final,columns=categorical,drop_first=True)


# In[46]:


main_dt_final.head()


# In[50]:


# Lets see how many columns we have now
print(len(main_dt_final.columns))
main_dt_final.shape


# In[51]:


#lets prepare the dataset for machine learning
from sklearn.model_selection import train_test_split


# In[52]:


# we remove the target variable 
x=main_dt_final.drop(['left'],axis=1).values
# we take the target variable in y
y=main_dt_final['left'].values


# In[53]:


# split the data in to a 70:30 Train:Test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[54]:


#Normalize the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[55]:


df_train=pd.DataFrame(x_train)
df_train.head()


# In[56]:


df_train.describe()


# # Let's train a Logistic Regression Model

# In[57]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

model=LogisticRegression()
model.fit(x_train,y_train)

predictions=model.predict(x_test)

print(" Accuracy {0:.2f}%".format(100*accuracy_score(predictions,y_test)))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# # Lets try a random forest clasifier

# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

model1=RandomForestClassifier()
model1.fit(x_train,y_train)

predictions1=model1.predict(x_test)

print(" Accuracy {0:.2f}%".format(100*accuracy_score(predictions1,y_test)))
print(confusion_matrix(y_test,predictions1))
print(classification_report(y_test,predictions1))


# # Lets try Deep Learning whether it makes a difference from random forest

# In[64]:


import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model2=Sequential()

model2.add(Dense(9,kernel_initializer='uniform',activation='relu',input_dim=18))
model2.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

model2.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])


# In[65]:


#display model summary and show the parameters
model2.summary()


# In[66]:


#start training our classifier model
batch_size=10
epochs= 25

history= model2.fit(x_train,y_train,
                   batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
score=model2.evaluate(x_test,y_test,verbose=0)
print('test loss:',score[0])
print('test Accuracy:',score[1])


# In[67]:


# plotting our loss charts

import matplotlib.pyplot as plt

history_dict=history.history

loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)

line1=plt.plot(epochs,val_loss_values,label='validation/testloss')
line2=plt.plot(epochs,loss_values,label='training loss')
plt.setp(line1,linewidth=2.0,marker='+',markersize=10.0)
plt.setp(line2,linewidth=2.0,marker='4',markersize=10.0)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()


# In[71]:


# plotting our accuracy charts
import matplotlib.pyplot as plt

history_dict=history.history

acc_values=history_dict['acc']
val_acc_values=history_dict['val_acc']
epochs=range(1,len(loss_values)+1)

line1=plt.plot(epochs,val_acc_values,label='validation/test accuracy')
line2=plt.plot(epochs,acc_values,label='training accuracy')
plt.setp(line1,linewidth=2.0,marker='+',markersize=10.0)
plt.setp(line2,linewidth=2.0,marker='4',markersize=10.0)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid(True)
plt.legend()
plt.show()


# # Displaying the classification Report and confusion matrix

# In[74]:


predictions2=model2.predict(x_test)
predictions2=predictions2>0.5


# In[75]:


print(" Accuracy {0:.2f}%".format(100*accuracy_score(predictions2,y_test)))
print(confusion_matrix(y_test,predictions2))
print(classification_report(y_test,predictions2))


# # Lets try a deeper model

# In[79]:


from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

model3=Sequential()

#Hidden layer 1
model3.add(Dense(100,activation='relu',input_dim=18,kernel_regularizer=l2(0.01)))
model3.add(Dropout(0.3,noise_shape=None,seed=None))

#Hidden Layer 2
model3.add(Dense(100,activation='relu',input_dim=18,kernel_regularizer=l2(0.01)))
model3.add(Dropout(0.3,noise_shape=None,seed=None))

model3.add(Dense(1,activation='sigmoid'))

model3.summary()


# In[80]:


model3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[81]:


#training our deeper model
batch_size=35
epochs= 25

history1= model3.fit(x_train,y_train,
                   batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
score=model3.evaluate(x_test,y_test,verbose=0)
print('test loss:',score[0])
print('test Accuracy:',score[1])


# In[82]:


predictions3=model3.predict(x_test)
predictions3=predictions3>0.5


print(" Accuracy {0:.2f}%".format(100*accuracy_score(predictions3,y_test)))
print(confusion_matrix(y_test,predictions3))
print(classification_report(y_test,predictions3))


# In[87]:


# Lets just check the most important features in the dataset as per Random forest classifier

feature_importance=pd.DataFrame(model1.feature_importances_,index=pd.DataFrame(x_test).columns,
                                columns=['importance']).sort_values('importance',ascending=False)

feature_importance


# In[ ]:




