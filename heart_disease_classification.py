## PROBLEM DEFINITION : given a patient's clinical parameters, can we predict if one has heart disease
## DATA: Cleveland data from UCI ML repository
## EVALUATION: >95 % accuracy
## FEATURES:  13 and 1 target

## COLLECTING TOOLS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, \
    plot_roc_curve

## DATA INPUT
df = pd.read_csv('heart-disease.csv')
# print(df.shape)
# print(df['target'].value_counts())
# print(df.info)
# print(df.isna().sum())
# print(df.describe())
# print(df.sex.value_counts())
# print(pd.crosstab(df.sex, df.target))

# pd.crosstab(df.sex, df.target).plot(kind='bar', figsize= (3,3))
# plt.title('Heart Disease v/s Sex')
# plt.xlabel('0 = No disease, 1 = Disease')
# plt.ylabel('Number')
# plt.legend(['Female', 'Male'])
# plt.xticks(rotation=0)
# plt.show();

## AGE AND MAX HEART RATE V/S HEART DISEASE
# plt.figure(figsize=(10,6))
# plt.scatter(df['age'][df['target'] == 0],
#             df['thalach'][df['target'] == 0],
#             c = 'salmon')
# plt.scatter(df['age'][df['target'] == 1],
#             df['thalach'][df['target'] == 1],
#             c = 'blue')
#
# plt.title('AGE AND MAX HEART RATE V/S HEART DISEASE')
# plt.xlabel('Age')
# plt.ylabel('Max Heart Rate')
# plt.legend(['No Disease', 'Disease'])
# plt.show()

## CHEST PAIN V/S HEART DISEASE
# pd.crosstab(df.cp, df.target)
# pd.crosstab(df.cp, df.target).plot(kind= 'bar', figsize= (10,6), color= ['salmon','blue'])
# plt.title('CHEST PAIN V/S HEART DISEASE')
# plt.xlabel('Chest Pain Level')
# plt.ylabel('Number')
# plt.legend(['No disease', 'Disease'])
# plt.xticks(rotation= 0)
# plt.show();

## FOR CORRELATION
# corr_matrix = df.corr()
# fig, ax = plt.subplots(figsize= (15,10))
# ax = sns.heatmap(corr_matrix, annot= True, linewidths=0.5, fmt='.2f', cmap='YlGnBu')
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom+0.5,top-0.5)
# plt.show()

## MODELLING
x = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
models = {'Logistic Regression': LogisticRegression(),
          'KNN': KNeighborsClassifier(),
          'RandomForest': RandomForestClassifier()}


def fit_score(models, x_train, x_test, y_train, y_test):
    np.random.seed(31)
    model_score = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        model_score[name] = model.score(x_test, y_test)
    return model_score


model_scores = fit_score(models=models, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
# print(model_scores)

# model_compare = pd.DataFrame(model_scores, index= ['accuracy'])
# model_compare.T.plot.bar()
# plt.xticks(rotation= 0)
# plt.show();

## IMPROVING MODEL
# tuning by hand

# train_scores = []
# test_scores = []
# neighbors = range(1, 11)
# knn = KNeighborsClassifier()

# for i in neighbors:
#     knn.set_params(n_neighbors= i)
#     knn.fit(x_train, y_train)
#     train_scores.append(knn.score(x_train, y_train))
#     test_scores.append(knn.score(x_test, y_test))
#
# print(test_scores, train_scores)

# plt.plot(neighbors, train_scores, label='Train')
# plt.plot(neighbors, test_scores, label='Test')
# plt.xlabel('Neighbors')
# plt.ylabel('Score')
# plt.legend()
# plt.show()
#
# print(f'max KNN on test : {max(test_scores) * 100:.2f} %')

## Randomized Search CV
# logistic regression

# log_grid = {'C': np.logspace(-4,4,20),
#             'solver': ['liblinear']}
#
# np.random.seed(31)
# rs_log_reg = RandomizedSearchCV(LogisticRegression(max_iter=100),
#                                 param_distributions= log_grid,
#                                 cv= 5,
#                                 n_iter= 20,
#                                 verbose= True)
#
# rs_log_reg.fit(x_train,y_train)
# print(rs_log_reg.best_params_)
# print(rs_log_reg.score(x_test,y_test))

# random forest classifier

# np.random.seed(31)
# rf_grid= {'n_estimators': np.arange(10,100,50),
#           'max_depth': [None, 3, 5, 10],
#           'min_samples_split': np.arange(2, 20, 2),
#           'min_samples_leaf': np.arange(1,20,2)}
# rs_rf = RandomizedSearchCV(RandomForestClassifier(),
#                            param_distributions=rf_grid,
#                            cv= 5,
#                            n_iter= 20,
#                            verbose= True)
#
# rs_rf.fit(x_train,y_train)
# print(rs_rf.best_params_)
# print(rs_rf.score(x_test,y_test))

## Grid Search CV
# log_reg_grid= {'C': np.logspace(-4,4,30),
#                'solver': ['liblinear']}
# gs_log_reg= GridSearchCV(LogisticRegression(),
#                          param_grid=log_reg_grid,
#                          cv= 5,
#                          verbose= True)
#
# gs_log_reg.fit(x_train,y_train)
# print(gs_log_reg.best_params_)
# print(gs_log_reg.score(x_test,y_test))
#
# y_preds= gs_log_reg.predict(x_test)
# plot_roc_curve(gs_log_reg, x_test, y_test)
# print(confusion_matrix(y_test,y_preds))
#
# def plot_conf_mat(y_test, y_preds):
#     fig,ax = plt.subplots(figsize= (3,3))
#     ax= sns.heatmap(confusion_matrix(y_test,y_preds),
#                     annot=True,
#                     cbar= False)
#     plt.xlabel('True Label')
#     plt.ylabel('Predicted Label')
#     bottom,top = ax.get_ylim()
#     ax.set_ylim(bottom+0.5, top-0.5)
#     plt.show()
#
# plot_conf_mat(y_test,y_preds)
# print(gs_log_reg.best_params_)
# classification matrix
# print(classification_report(y_test,y_preds))

# cross val score

# clf = LogisticRegression(C= 2.592943797404667,
#                          solver= 'liblinear')
# cv_acc = cross_val_score(clf,x,y, cv= 5, scoring= 'accuracy')
# print(np.mean(cv_acc))
# cv_prec = cross_val_score(clf,x,y,cv= 5, scoring= 'precision')
# print(np.mean(cv_prec))
# cv_recall = cross_val_score(clf,x,y,cv= 5, scoring= 'recall')
# print(np.mean(cv_prec))
# cv_f1 = cross_val_score(clf,x,y,cv= 5, scoring= 'f1')
# print(np.mean(cv_prec))
#
# cv_metrics= pd.DataFrame({'Accuracy': cv_acc,
#                           'Precision': cv_prec,
#                           'Recall': cv_recall,
#                           'f1': cv_f1})
# cv_metrics.T.plot.bar(title = 'CV classification metrics', legend= False)
# plt.show()

## FEATURE IMPORTANCE
# clf = LogisticRegression(C= 2.592943797404667,
#                          solver= 'liblinear')
# clf.fit(x_train,y_train)
# print(clf.coef_)
# feature_dict= dict(zip(df.columns, list(clf.coef_[0])))
# feature_df= pd.DataFrame(feature_dict, index=[0])
# feature_df.T.plot.bar(title= 'Feature Importance',
#                       legend= False)
# plt.show();