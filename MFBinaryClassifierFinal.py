#!/usr/bin/env python
# coding: utf-8

# ### Packages

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import graphviz
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn import svm


# # Data Exploration & Visualization

# In[2]:


#import data into dataframe
voice_df = pd.read_csv("voice.csv")
#preview data
voice_df.head()


# In[3]:



#all mean frequency 
sns.set(style="darkgrid")
sns.distplot( voice_df["meanfreq"] , color="green", label="All")
plt.legend()
plt.show()


# In[4]:


#sort df by gender
gender = voice_df.set_index(['label'])
#df for females
female_df = gender.loc['female']
#def for males
male_df = gender.loc['male']


# In[5]:


#female hist mean freq
sns.distplot( female_df["meanfreq"] , color="red", label="Female")
plt.legend()
plt.show()


# In[6]:


#male hist mean freq
sns.distplot( male_df["meanfreq"] , color="skyblue", label="Male")
plt.legend()
plt.show()


# In[7]:


#overlay two hists
sns.distplot( female_df["meanfreq"] , color="red", label="Female")
sns.distplot( male_df["meanfreq"] , color="skyblue", label="Male")

plt.legend()

plt.show()


# In[8]:


count_row = voice_df.shape[0]  # gives number of row count
male_count = male_df.shape[0]
female_count = female_df.shape[0]
print count_row, male_count, female_count
#make sure we have an equal ratio 
#male ratio
print("male ratio: ")
print(male_count/float(count_row))
#female ratio
print("female ratio: ")
print(female_count/float(count_row))


# ## PCA

# In[9]:


# ZScore data for PCA Analysis
#Convert labels into integer classes
y = voice_df[['label']]
#covert to dumm variable,1 for female, 0 for male.
y = [1 if each == "female" else 0 for each in y.label]

#Scale features
X=preprocessing.scale(voice_df.drop(columns=['label']))

#PCA Analysis to visualize data in a low-dimension
y_arr=np.asarray(y)

#TWO Component PCA
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['red', 'blue']
lw = 2

for color, i, target_name in zip(colors, [0, 1], ['female','male']):
    plt.scatter(X_r[y_arr == i, 0], X_r[y_arr == i, 1], color=color, alpha=.5, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('2 Component PCA of Voice dataset')



#THREE Component PCA
pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, i, target_name in zip(colors, [0, 1], ['female','male']):
    
    ax.scatter(X_r[y_arr == i, 0], X_r[y_arr == i, 1],X_r[y_arr == i, 2], c=c, label=target_name)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3 Component PCA of Voice dataset')
ax.legend(loc='best', shadow=False, scatterpoints=1)

#ax.view_init(60, 60)
plt.show()


# # Data Preparation
# The data used to train and test the algorithms was normalized using MinMax normalization limiting the range of values to (0,1].

# In[10]:


#try with all features
X = voice_df.drop('label', axis=1) 

#get target labels. 
y = voice_df[['label']]
#covert to dumm variable,1 for female, 0 for male.
y = [1 if each == "female" else 0 for each in y.label]


#normalize the features using min-max
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)
                         

#preview
X.head()


# In[11]:


#80/20 split for train and test dataset. X contains the features, y contains the target labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Train Neural Network

# In[12]:


#build model
NN_build = MLPClassifier(solver='lbfgs', alpha=1e-2, random_state=1)
#fit model
NN = NN_build.fit(X_train, y_train) 
#view hyperparameters
print(NN)


# # Evaluate Neural Network

# In[13]:


#get accuracy and generate confusion matrix

#get predictions
predictions = NN.predict(X_test)
#generate confusion matrix 
cm = confusion_matrix(y_test,predictions)
#visualize confusion matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
#get accuracy 
print("accuracy:",accuracy_score(y_test, predictions))


# # Improve Neural Network

# Optimize by cross-validated grid-search over a parameter grid.

# In[14]:


#try different hyperparameters 
parameters = {
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs'], 
    'max_iter': [300], 
    'alpha': 10.0 ** -np.arange(1, 7), 
    'hidden_layer_sizes': [(6),(5,2), (10,10,10), (20)],
    'random_state':[1]
}
NN_grid = GridSearchCV(MLPClassifier(), parameters, cv = 5)


# In[15]:


#fit and retrieve best hyperparameters
NN_grid.fit(X_train, y_train)
print(NN_grid.best_params_)
print("Best score: %0.4f" % NN_grid.best_score_)


# Build new neural network with optimized hyperparameters

# In[16]:


#build new model
NN_new_build = MLPClassifier(activation = 'tanh',
                             solver='lbfgs', 
                             alpha=0.1, 
                             hidden_layer_sizes=(20), 
                             max_iter = 300,  
                             random_state=1)
#fit training data 
NN_new = NN_new_build.fit(X_train, y_train) 
#view hyperparameters
print(NN_new)


# # Evaluate New Neural Network

# In[17]:


#get predictions
predictions_new = NN_new.predict(X_test)
#generate confusion matrix 
cm = confusion_matrix(y_test,predictions_new)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
#get accuracy 
print("accuracy:",accuracy_score(y_test, predictions_new))


# In[18]:


#generate ROC
#get probabilities for positives
probs = NN_new.predict_proba(X_test)
probs = probs[:, 1]
#get auc score
auc = roc_auc_score(y_test, probs)

#plot ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.3f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network ROC')
plt.legend(loc="lower right")
plt.legend(loc="lower right")
# show the plot
plt.show()


# In[19]:


#Compare Empirical MCR to CV MCR look for overfitting
#Empirical MCR
predictions = NN_new.predict(X)
MCR = 1 - accuracy_score(y, predictions)
print("Empirical MCR: ",round( MCR, 3))

#CV MCR
scores = cross_val_score(NN_new_build, X, y, cv=10)
CV_MCR = 1 - scores.mean()
print("CV MCR: ", round(MCR,3))


# # Visualize NN Results in Low-Dimension

# In[20]:


results=NN_new.predict(X_test) == y_test
results=np.asarray(results)

#TWO Component PCA
pca = PCA(n_components=2)
X_r = pca.fit(X_test).transform(X_test)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['green', 'red']
lw = 2

for color, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    plt.scatter(X_r[results == i, 0], X_r[results == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('2 Component PCA of NN Classification Results')



#THREE Component PCA
pca = PCA(n_components=3)
X_r = pca.fit(X_test).transform(X_test)

# Percentage of variance explained for each components
print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    
    ax.scatter(X_r[results == i, 0], X_r[results == i, 1],X_r[results == i, 2], c=c, )

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3 Component PCA of NN Classification Results')

plt.show()


# # Train Logistic Regression

# In[21]:


logreg_build = LogisticRegression(solver = 'lbfgs')
logreg = logreg_build.fit(X_train,y_train) 
#view hyperparameters 
logreg


# # Evaluate Logistic Regression

# In[22]:


#get accuracy and generate confusion matrix

#get predictions
predictions = logreg.predict(X_test)
cm = confusion_matrix(y_test,predictions)
#visualize confusion matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
print("accuracy:", accuracy_score(y_test, predictions))


# # Improve Logistic Regression 

# Optimize by cross-validated grid-search over a parameter grid.

# In[23]:


parameters ={
    'C':np.logspace(-3,3,7), 
    'solver':['lbfgs'], 
    'max_iter': [300]
}
#try different parameters 
logreg_grid = GridSearchCV(LogisticRegression(), parameters ,cv = 5)
#fit and retrieve best hyperparameters
logreg_grid.fit(X_train, y_train)
#fit and retrieve best hyperparameters
print(logreg_grid.best_params_)
print("Best score: %0.4f" % logreg_grid.best_score_)


# Build new logistic regression model with optimized hyperparameters

# In[24]:


logreg_new_build = LogisticRegression(C = 100.0, 
                                      max_iter = 300, 
                                      solver ='lbfgs')
#fit training data 
logreg_new = logreg_new_build.fit(X_train, y_train) 
#view hyperparameters
logreg_new


# # Evaluate New Logistic Regression Model

# In[25]:


#get predictions
predictions = logreg_new.predict(X_test)
cm = confusion_matrix(y_test,predictions)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
print("accuracy:", accuracy_score(y_test, predictions))


# In[26]:


#generate ROC

#get probabilities for positives
probs = logreg_new.predict_proba(X_test)
probs = probs[:, 1]
#get auc score
auc = roc_auc_score(y_test, probs)

#plot ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker = '.', label='ROC curve (area = %0.3f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC')
plt.legend(loc="lower right")
plt.legend(loc="lower right")
# show the plot
plt.show()


# In[27]:


#Empirical MCR
predictions = logreg_new.predict(X)
MCR = 1 - accuracy_score(y, predictions)
print("Empirical MCR: ",round( MCR, 3))

#CV MCR
scores = cross_val_score(logreg_new_build, X, y, cv=10)
CV_MCR = 1 - scores.mean()
print("CV MCR: ", round(MCR,3))


# # Visualize Logist Regression Results in Low-Dimension

# In[28]:


results=logreg_new.predict(X_test) == y_test 
results=np.asarray(results)

#TWO Component PCA
pca = PCA(n_components=2)
X_r = pca.fit(X_test).transform(X_test)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['green', 'red']
lw = 2

for color, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    plt.scatter(X_r[results == i, 0], X_r[results == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('2 Component PCA of LR Classification Results')



#THREE Component PCA
pca = PCA(n_components=3)
X_r = pca.fit(X_test).transform(X_test)

# Percentage of variance explained for each components
print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    
    ax.scatter(X_r[results == i, 0], X_r[results == i, 1],X_r[results == i, 2], c=c, )

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3 Component PCA of LR Classification Results')

plt.show()


# # Train Naive Bayes

# In[29]:


# Fit NaiveBayes model to our data
# build model
gb_mdl = GaussianNB()
# fit model
gb_mdl.fit(X_train, y_train)
# view hyperparameters
gb_mdl 


# # Evaluate Naive Bayes Model

# In[30]:


# get accuracy and generate confusion matrix

# get predictions
predictions = gb_mdl.predict(X_test)
# generate confusion matrix
cm = confusion_matrix(y_test, predictions)
#visualize confusion matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
#get accuracy 
print("accuracy:",accuracy_score(y_test, predictions))


# In[31]:


# ROC Curve to check performance of all NaiveBayes 
#generate ROC

#get probabilities for positives
probs = gb_mdl.predict_proba(X_test)
probs = probs[:, 1]
#get auc score
auc = roc_auc_score(y_test, probs)

#plot ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.3f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC')
plt.legend(loc="lower right")
plt.legend(loc="lower right")
# show the plot
plt.show()


# In[32]:


#Empirical MCR
predictions = gb_mdl.predict(X)
MCR = 1 - accuracy_score(y, predictions)
print("Empirical MCR: ",round( MCR, 3))

#CV MCR
scores = cross_val_score(gb_mdl, X, y, cv=10)
CV_MCR = 1 - scores.mean()
print("CV MCR: ", round(MCR,3))


# # Visualize Naive Bayes Results in Low-Dimension

# In[33]:


results=gb_mdl.predict(X_test) == y_test 
results=np.asarray(results)

#TWO Component PCA
pca = PCA(n_components=2)
X_r = pca.fit(X_test).transform(X_test)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['green', 'red']
lw = 2

for color, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    plt.scatter(X_r[results == i, 0], X_r[results == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('2 Component PCA of GNB Classification Results')



#THREE Component PCA
pca = PCA(n_components=3)
X_r = pca.fit(X_test).transform(X_test)

# Percentage of variance explained for each components
print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    
    ax.scatter(X_r[results == i, 0], X_r[results == i, 1],X_r[results == i, 2], c=c, )

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3 Component PCA of GNB Classification Results')

plt.show()


# # Train XGBoost Model

# In[34]:



# build model
xgb_mdl = XGBClassifier()
# fit model
xgb_mdl.fit(X_train,y_train)
# view hyperparameters
xgb_mdl


# # Evaluate XGBoost

# In[35]:



#get accuracy and generate confusion matrix

#get predictions
predictions = xgb_mdl.predict(X_test)
#generate confusion matrix 
cm = confusion_matrix(y_test,predictions)
#visualize confusion matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
#get accuracy 
print("accuracy:",accuracy_score(y_test, predictions))


# # Improve XGBoost

# Optimize by cross-validated grid-search over a parameter grid

# In[72]:



# Let's do a grid search with parameters around default since XGBoost performs well
# Let's do a grid search with parameters around default since XGBoost performs well
parameters = {
    'learning_rate': [i/10.0 for i in range(2,4)],
    'gamma': [i/10.0 for i in range(0,2)],
    'max_depth': range(5,7),
    'min_child_weight': range(1,3),
    'max_delta_step': range(0,2),
    'subsample': [i/10.0 for i in range(8,10)],
    'colsample_bytree': [i/10.0 for i in range(8,10)],
    'colsample_bylevel': [i/10.0 for i in range(8,10)]
}
grid_search = GridSearchCV(xgb_mdl, parameters)


# In[73]:



# fit and retrieve best hyperparameters
grid_search.fit(X_train,y_train)
grid_search.best_score_, grid_search.best_params_
print(grid_search.best_params_)
print("Best score: %0.4f" % grid_search.best_score_)


# Build new XGBoost with optimized hyperparameters

# In[74]:



# Final Model Optimized
xgb_opt = XGBClassifier(colsample_bylevel=0.8,
                        colsample_bytree=0.8,
                        gamma=0.0,
                        learning_rate=0.2,
                        max_delta_step=1,
                        max_depth=5,
                        min_child_weight=1,
                        subsample=0.8)
xgb_opt.fit(X_train,y_train)


# # Evaluate new XGBoost Model

# In[75]:



#get predictions
predictions_new = xgb_opt.predict(X_test)
#generate confusion matrix 
cm = confusion_matrix(y_test,predictions_new)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
#get accuracy 
print("accuracy:",accuracy_score(y_test, predictions_new))


# In[76]:



#generate ROC

#get probabilities for positives
probs = xgb_opt.predict_proba(X_test)
probs = probs[:, 1]
#get auc score
auc = roc_auc_score(y_test, probs)

#plot ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.3f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC')
plt.legend(loc="lower right")
plt.legend(loc="lower right")
# show the plot
plt.show()


# In[77]:



#Compare Empirical MCR to CV MCR look for overfitting
#Empirical MCR
predictions = xgb_opt.predict(X)
MCR = 1 - accuracy_score(y, predictions)
print("Empirical MCR: ",round( MCR, 3))

#CV MCR
scores = cross_val_score(xgb_opt, X, y, cv=10)
CV_MCR = 1 - scores.mean()
print("CV MCR: ", round(MCR,3))


# # Visualize XGBoost Results in Low-Dimension

# In[78]:




results=xgb_opt.predict(X_test) == y_test 
results=np.asarray(results)

#TWO Component PCA
pca = PCA(n_components=2)
X_r = pca.fit(X_test).transform(X_test)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['green', 'red']
lw = 2

for color, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    plt.scatter(X_r[results == i, 0], X_r[results == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('2 Component PCA of XGB Classification Results')



#THREE Component PCA
pca = PCA(n_components=3)
X_r = pca.fit(X_test).transform(X_test)

# Percentage of variance explained for each components
print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    
    ax.scatter(X_r[results == i, 0], X_r[results == i, 1],X_r[results == i, 2], c=c, )

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3 Component PCA of XGB Classification Results')

plt.show()


# # Train Decision Tree Model

# In[5]:


# build model
dt_mdl = tree.DecisionTreeClassifier(max_depth=3) #96.5%
#fit model
dt_mdl = dt_mdl.fit(X_train, y_train) 
#view model details
print(dt_mdl)


# # Evaluate Decision Tree

# In[44]:



#get predictions
predictions = dt_mdl.predict(X_test)
#generate confusion matrix 
cm = confusion_matrix(y_test,predictions)
#visualize confusion matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
#get accuracy 
print("accuracy:",accuracy_score(y_test, predictions))


# In[4]:


# plot decision tree graph
dot_data = tree.export_graphviz(dt_mdl, 
                                out_file=None,
                                feature_names=list(X_train),
                                class_names='label',
                                filled=True, 
                                rounded=True, 
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# In[46]:


#generate ROC

#get probabilities for positives
probs = dt_mdl.predict_proba(X_test)
probs = probs[:, 1]
#get auc score
auc = roc_auc_score(y_test, probs)

#plot ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.3f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree ROC')
plt.legend(loc="lower right")
plt.legend(loc="lower right")
# show the plot
plt.show()


# In[47]:


predictions = dt_mdl.predict(X)
MCR = 1 - accuracy_score(y, predictions)
print("Empirical MCR: ",round( MCR, 3))

#CV MCR
scores = cross_val_score(dt_mdl, X, y, cv=10)
CV_MCR = 1 - scores.mean()
print("CV MCR: ", round(MCR,3))


# # Visualize Decision Tree Results in Low-Dimension

# In[48]:


results=dt_mdl.predict(X_test) == y_test 
results=np.asarray(results)

#TWO Component PCA
pca = PCA(n_components=2)
X_r = pca.fit(X_test).transform(X_test)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['green', 'red']
lw = 2

for color, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    plt.scatter(X_r[results == i, 0], X_r[results == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('2 Component PCA of DT Classification Results')



#THREE Component PCA
pca = PCA(n_components=3)
X_r = pca.fit(X_test).transform(X_test)

# Percentage of variance explained for each components
print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    
    ax.scatter(X_r[results == i, 0], X_r[results == i, 1],X_r[results == i, 2], c=c, )

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3 Component PCA of DT Classification Results')

plt.show()


# # Train SVM Model

# In[49]:


# build model
svm_mdl = svm.SVC(probability=True)
# fit model
svm_mdl.fit(X_train, y_train)
# view hyperparameters
print(svm_mdl)


# # Evaluate SVM Model

# In[50]:


#get accuracy and generate confusion matrix

#get predictions
predictions = svm_mdl.predict(X_test)
#generate confusion matrix 
cm = confusion_matrix(y_test,predictions)
#visualize confusion matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
#get accuracy 
print("accuracy:",accuracy_score(y_test, predictions))


# # Improve SVM Model

# In[51]:


param_grid = [
  {
      'C': [1], 
      'degree': [0,1,2,3,4,5,6], 
      'kernel': ['poly']
  },
  {
      'C': [1, 10, 100, 1000], 
      'gamma': [0.001, 0.0001], 
      'kernel': ['rbf']
  }
 ]

svm_opt = svm.SVC(probability=True) 
svm_grid = GridSearchCV(svm_opt, param_grid, cv=10)


# In[52]:


#fit and retrieve best hyperparameters
svm_grid.fit(X_train, y_train)
print(svm_grid.best_params_)
print("Best score: %0.4f" % svm_grid.best_score_)


# In[53]:


# build new model
svm_opt = svm.SVC(kernel='rbf',
                 C=1000,
                 gamma=0.001,
                 probability=True)
# fit training data
svm_opt.fit(X_train,y_train)
# view hyperparameters
print(svm_opt)


# # Evaluate Optimized SVM Model

# In[54]:


#get predictions
predictions_new = svm_opt.predict(X_test)
#generate confusion matrix 
cm = confusion_matrix(y_test,predictions_new)
#Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
#get accuracy 
print("accuracy:",accuracy_score(y_test, predictions_new))


# In[55]:



#generate ROC

#get probabilities for positives
probs = svm_opt.predict_proba(X_test)
probs = probs[:, 1]
#get auc score
auc = roc_auc_score(y_test, probs)

#plot ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.3f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC')
plt.legend(loc="lower right")
plt.legend(loc="lower right")
# show the plot
plt.show()


# In[56]:


predictions = svm_opt.predict(X)
MCR = 1 - accuracy_score(y, predictions)
print("Empirical MCR: ",round( MCR, 3))

#CV MCR
scores = cross_val_score(dt_mdl, X, y, cv=10)
CV_MCR = 1 - scores.mean()
print("CV MCR: ", round(MCR,3))


# ## Visualize SVM Results in Low Dimensions

# In[57]:


results=svm_opt.predict(X_test) == y_test 
results=np.asarray(results)

#TWO Component PCA
pca = PCA(n_components=2)
X_r = pca.fit(X_test).transform(X_test)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['green', 'red']
lw = 2

for color, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    plt.scatter(X_r[results == i, 0], X_r[results == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('2 Component PCA of SVM Classification Results')



#THREE Component PCA
pca = PCA(n_components=3)
X_r = pca.fit(X_test).transform(X_test)

# Percentage of variance explained for each components
print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, i, target_name in zip(colors, [1, 0], ['correct','incorrect']):
    
    ax.scatter(X_r[results == i, 0], X_r[results == i, 1],X_r[results == i, 2], c=c, )

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3 Component PCA of SVM Classification Results')

plt.show()


# # Model Comparison Plot

# In[58]:



# Model Comparison Plot
seed = 7
models = []
models.append(('NN',NN_new))
models.append(('LR',logreg_new))
models.append(('GNB',gb_mdl))
models.append(('XGB',xgb_opt))
models.append(('DT',dt_mdl))
models.append(('SVM',svm_opt))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
sns.boxplot(x=names,y=results)
plt.title('Model Comparison with 10CV')
plt.ylabel('Mean Accuracy')
plt.show()

