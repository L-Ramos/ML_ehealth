# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:08:45 2020

@author: laramos
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
import pandas as pd
import os
import numpy as np

import data_clean as dt
import methods as mt
import utils as ut

args = dict()
args['random_state'] = 1         
args['undersample'] = 'B'
args['n_jobs'] = -2
args['outter_splits'] = 10
args['verbose'] = 1
args['n_iter_search'] = 25
args['opt_measure'] = 'roc_auc'
args['cv_plits'] = 3
args['class_weight'] = 'balanced'

PATH_RESULTS = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results"
PATH_DATA = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\feature_datasets"
# time_hours_list = [72, np.inf]
# program_list = ['Alcohol','Cannabis', 'Smoking']
# feature_type_list = ['safe','all']

time_hours_list = [72]
program_list = ['Alcohol','Cannabis', 'Smoking']
feature_type_list = ['safe']
experiment = 'exp1'
drop_afspraak = False

for program in program_list:
    for time_hours in time_hours_list:
        for feature_type in feature_type_list:
        
            path_write = os.path.join(PATH_RESULTS,program+str(time_hours)+"_"+feature_type+"_"+ experiment)
    
            if not os.path.exists(path_write):
                os.mkdir(path_write)
        
            df_temp = pd.read_csv(os.path.join(PATH_DATA,program+str(time_hours)+'False.csv'))
            var_list = pd.read_csv(os.path.join(PATH_DATA,'var_list.csv'))
            
            args, feats_use  = dt.get_features(args, feature_type,df_temp.columns,var_list)
            
            X, y  = dt.clean_data(df_temp,args,experiment,drop_afspraak,feats_use)
            
            dt.table_one(path_write,X,y,args,time_hours)
            
            kf = KFold(n_splits = args['outter_splits'], random_state=1, shuffle=True)
            
            rfc_m, _, _, _, _ = ut.create_measures(args['outter_splits'])
            
            for fold,(train_index, test_index) in enumerate(kf.split(X)):
                X_train,y_train = X.iloc[train_index,:], y[train_index]
                X_test,y_test = X.iloc[test_index,:], y[test_index]
                args['current_iteration'] = fold
                args['pos_weights'] = (y_train.shape[0]-sum(y_train))/sum(y_train)
                grid_rfc = mt.RandomForest_CLF(args,X_train,y_train,X_test,y_test,rfc_m,test_index)
                #grid_lr = mt.LogisticRegression_CLF(args,X_train,y_train,X_test,y_test,lr_m,test_index)
            
            
            names = ['RFC']
            meas = [rfc_m]
            mt.print_results_excel(meas,names,path_write)
            mt.plot_shap(X,args,path_write, rfc_m.feat_names)
            
            print("Auc: %0.2f, Sens: %0.2f, Spec: %0.2f, PPV: %0.2f, NPV: %0.2f"%(rfc_m.auc[1],rfc_m.sens[1],rfc_m.spec[1],rfc_m.ppv[1],rfc_m.npv[1]))
      

# import shap
# shap_values = rfc_m.shap_values[0][1]
# shap.summary_plot(shap_values, X_test, plot_type="bar",show=True,auto_size_plot=(20,10))
# shap.summary_plot(shap_values, X_test,show=True,auto_size_plot=(20,10))

#c = mt.LogisticRegression_CLF(args,f_train,y_train,f_test,y_test,meas)
#c = mt.SupportVectorMachine_CLF(args,f_train,y_train,f_test,y_test,meas)

# Here I'm just checking stuff, below I plot box plots
# import seaborn as sns

# df_temp_plot = df_temp.copy()
# df_temp_plot['Phase'] = df_temp_plot['Phase']>1
# df_temp_plot = df_temp_plot[['Pros Short', 'Pros Long', 'Cons Short', 'Cons Long', 'Start video','Afspraken maken', 'Jouw afspraken','Phase']]

# sns.boxplot(data=df_temp_plot[['Pros Short', 'Pros Long', 'Cons Short', 'Cons Long']],orient='h')
# df_temp_plot.boxplot(column = ['Pros Short', 'Jouw afspraken'], by='Phase')
