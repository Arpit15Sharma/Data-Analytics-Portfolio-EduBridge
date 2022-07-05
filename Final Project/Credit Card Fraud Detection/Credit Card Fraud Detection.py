#!/usr/bin/env python
# coding: utf-8

# # Dataset Information

# The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

# # Import modules

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')


# # <b>Loading the dataset

# In[2]:


data = pd.read_csv('creditcard.csv')


# ## Data exploration

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.dtypes


# In[7]:


len(data)


# In[8]:


data.isnull().sum()


# In[9]:


data.describe()


# # Histograms

# In[10]:


fig = plt.figure(figsize=(15, 20))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(data.shape[1]):
    plt.subplot(8, 4, i + 1)
    f = plt.gca()
    f.set_title(data.columns.values[i])

    vals = np.size(data.iloc[:, i].unique())
    if vals >= 100:
        vals = 100                                  
    
    plt.hist(data.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# - we can observe a large prevalence of Class 0 (non fraudulent).

# In[11]:


print('Number of fraudulent transactions = %d or %d per 100,000 transactions in the dataset'
      %(len(data[data.Class==1]), len(data[data.Class==1])/len(data)*100000))


# ## Linear Correlation with Response Variable (Note: Models like RandomForest are not linear)

# In[12]:


data2 = data.drop(columns = ['Class']) 
data2.corrwith(data.Class).plot.bar(
        figsize = (20, 10), title = "Correlation with Class Fraudulent or Not", fontsize = 15,
        rot = 45, grid = True)
plt.show()


# ## Pre-processing

# In[13]:


from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))  
data = data.drop(['Amount'],axis=1)


# In[14]:


data.head()


# In[15]:


data = data.drop(['Time'],axis=1)
data.head()


# In[16]:


X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']


# In[17]:


y.head()


# ## Data Split

# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# ## Random Forest

# In[21]:


from sklearn.ensemble import RandomForestClassifier


# In[22]:


random_forest = RandomForestClassifier(n_estimators=100)


# In[23]:


# Pandas Series.ravel() function returns the flattened underlying data as an ndarray.
random_forest.fit(X_train,y_train.values.ravel())    # np.ravel() Return a contiguous flattened array


# In[24]:


start=time.time()
y_pred = random_forest.predict(X_test)
end = time.time()
Time_Taken= end-start
Time_Taken


# In[25]:


random_forest.score(X_test,y_test)


# ## confusion matrix 

# In[26]:


import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# # Confusion matrix on the test dataset

# In[27]:


# Confusion matrix on the test dataset
cnf_matrix = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cnf_matrix,classes=[0,1])


# - while only 6 regular transactions are wrongly predicted as fraudulent, the model only detects 78% of the fraudulent transactions. As a consequence 33 fraudulent transactions are not detected (False Negatives).
# - Let's see if we can improve this performance with other machine learning / deep learning models in the rest of the notebook.

# In[28]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, plot_roc_curve
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('accuracy:%0.4f'%acc,'\tprecision:%0.4f'%prec,'\trecall:%0.4f'%rec,'\tF1-score:%0.4f'%f1,"Time Taken:",end-start)


# Let's store each model's performance in a dataframe for comparison purpose

# In[29]:


### Store results in dataframe for comparing various Models
results_testset = pd.DataFrame([['RandomForest', acc, 1-rec, rec, prec, f1,Time_Taken]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score','Time Taken'])
results_testset


# In[30]:


ROC_RF = plot_roc_curve(random_forest, X_test, y_test)
plt.show()


# We will run the models on the full dataset to check.

# In[31]:


# Confusion matrix on the whole dataset
y_pred = random_forest.predict(X)
cnf_matrix = confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])


# In[32]:


acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
print('accuracy:%0.4f'%acc,'\tprecision:%0.4f'%prec,'\trecall:%0.4f'%rec,'\tF1-score:%0.4f'%f1)


# In[33]:


results_fullset = pd.DataFrame([['RandomForest', acc, 1-rec, rec, prec, f1,Time_Taken]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score','Time Taken'])
results_fullset


# ## Decision trees

# In[34]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()


# In[35]:


decision_tree.fit(X_train,y_train.values.ravel())


# In[36]:


start=time.time()
y_pred = decision_tree.predict(X_test)
end=time.time()
time_dc=end-start
time_dc


# In[37]:


decision_tree.score(X_test,y_test)


# In[38]:


# Confusion matrix on the test dataset
cnf_matrix = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cnf_matrix,classes=[0,1])


# - The performance of the Decision Tree model is below the one using Random Forest. Let's check the performance indicators.

# In[39]:


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[40]:


### Store results in dataframe for comparing various Models
model_results = pd.DataFrame([['DecisionTree', acc, 1-rec, rec, prec, f1,time_dc]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score','Time Taken'])
results_testset = results_testset.append(model_results, ignore_index = True)
results_testset


# In[41]:


ROC_DT = plot_roc_curve(decision_tree, X_test, y_test)
plt.show()


# In[42]:


# Confusion matrix on the whole dataset
y_pred = decision_tree.predict(X)
cnf_matrix = confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])


# In[43]:


acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)


# In[44]:


model_results = pd.DataFrame([['DecisionTree', acc, 1-rec, rec, prec, f1,time_dc]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score','Time Taken'])
results_fullset = results_fullset.append(model_results, ignore_index = True)
results_fullset

