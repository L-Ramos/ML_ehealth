# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:08:45 2020

@author: laramos
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
import pandas as pd
import methods as mt
import utils as ut


df_temp = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python\features.csv")

all_features = ['Pros Short', 'Pros Long', 'Cons Short', 'Cons Long', 'Start video','Voor- en nadelen','Afspraken maken', 'Jouw afspraken','Number of Agreements', 'Total Agreement Length']
only_safe_features = ['Start video','Voor- en nadelen','Afspraken maken', 'Jouw afspraken']

X = df_temp[only_safe_features]
y = (df_temp['Phase']>1).astype('int32')
X['Jouw afspraken'] = X['Jouw afspraken'].fillna(0)
X['Afspraken maken'] = X['Afspraken maken'].fillna(0)
X['Start video'] = X['Start video'].fillna(0)
X['Voor- en nadelen'] = X['Voor- en nadelen'].fillna(0)
X = X.fillna(-1)

#X,y = load_breast_cancer(return_X_y = True, as_frame = True)

mask_cont = X.columns

kf = KFold(n_splits = 2, random_state=1, shuffle=True)


for fold,(train_index, test_index) in enumerate(kf.split(X)):
    X_train,y_train = X.iloc[train_index,:], y[train_index]
    X_test,y_test = X.iloc[test_index,:], y[test_index]
    
 
args = dict()
args['random_state'] = 1         
args['undersample'] = 'W'
args['n_jobs'] = -2
args['outter_splits'] = 2
args['verbose'] = 1
args['n_iter_search'] = 25
args['opt_measure'] = 'roc_auc'
args['cv_plits'] = 5
args['mask_cont'] = mask_cont
args['current_iteration'] = fold
args['class_weight'] = 'balanced'
args['pos_weights'] = (y_train.shape[0]-sum(y_train))/sum(y_train)

rfc_m, _, _, _, _ = ut.create_measures(args['outter_splits'])

grid_rfc = mt.RandomForest_CLF(args,X_train,y_train,X_test,y_test,rfc_m,test_index)

print("Auc: %0.2f, Sens: %0.2f, Spec: %0.2f, PPV: %0.2f, NPV: %0.2f"%(rfc_m.auc[1],rfc_m.sens[1],rfc_m.spec[1],rfc_m.ppv[1],rfc_m.npv[1]))

# Here I'm just checking stuff, below I plot box plots
import seaborn as sns

df_temp_plot = df_temp.copy()
df_temp_plot['Phase'] = df_temp_plot['Phase']>1
df_temp_plot = df_temp_plot[['Pros Short', 'Pros Long', 'Cons Short', 'Cons Long', 'Start video','Afspraken maken', 'Jouw afspraken','Phase']]

sns.boxplot(data=df_temp_plot[['Pros Short', 'Pros Long', 'Cons Short', 'Cons Long']],orient='h')
df_temp_plot.boxplot(column = ['Pros Short', 'Jouw afspraken'], by='Phase')


import shap
shap_values = rfc_m.shap_values[0][1]
shap.summary_plot(shap_values, X_test, plot_type="bar",show=True,auto_size_plot=(20,10))
shap.summary_plot(shap_values, X_test,show=True,auto_size_plot=(20,10))

#c = mt.LogisticRegression_CLF(args,f_train,y_train,f_test,y_test,meas)
#c = mt.SupportVectorMachine_CLF(args,f_train,y_train,f_test,y_test,meas)
