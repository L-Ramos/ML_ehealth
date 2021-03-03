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
args['undersample'] = 'W'
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
#program_list = ['Alcohol','Cannabis', 'Smoking']
# feature_type_list = ['safe','all']

time_hours_list = [72]
program_list = ['Alcohol','Cannabis', 'Smoking']
feature_type_list = ['safe']
experiment = 'exp3'
drop_afspraak = False
thresh_corr = 0.8
min_goalphase = [4,5,6]
min_goaldays = 7

for goal_phase in min_goalphase:
    for program in program_list:
        for time_hours in time_hours_list:
            for feature_type in feature_type_list:
            
                path_write = os.path.join(PATH_RESULTS,program+str(time_hours)+"_"+feature_type+"_min_days_"+str(min_goaldays)+"_min_phase_"+str(goal_phase)+experiment)
        
                if not os.path.exists(path_write):
                    os.mkdir(path_write)
            
                df_temp = pd.read_csv(os.path.join(PATH_DATA,program+str(time_hours)+'.csv'))
                var_list = pd.read_csv(os.path.join(PATH_DATA,'var_list.csv'))
                
                args, feats_use  = dt.get_features(args, feature_type,df_temp.columns,var_list)
                
                X, y  = dt.clean_data(df_temp,args,experiment,drop_afspraak,min_goaldays,goal_phase,feats_use)
                
                X,args = dt.correlation(X,thresh_corr,args) 
                            
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
      
        
#%%
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
     
path = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results\Alcohol72_safe_min_days_7_min_phase_6exp3\measures_RFC.pkl"

with open(path, 'rb') as f:
    meas = pickle.load(f)
    

array = [[np.sum(meas.tp),np.sum(meas.fp)],[np.sum(meas.fn),np.sum(meas.tn)]]
df_cm = pd.DataFrame(array, index = [i for i in ["Success","Drop-out"]],
                  columns = [i for i in ["Success","Drop-out"]])
plt.figure(figsize = (11,8))
sns.set(font_scale=2.0)
sns.heatmap(df_cm, annot=True,fmt = ".1f",cmap="Blues")    
        

#%% univariable analysis
import statsmodels.api as sm
import xlwt

book = xlwt.Workbook(encoding="utf-8")    
sheet1 = book.add_sheet("Sheet 1")

sheet1.write(0, 0, "Feature")
sheet1.write(0, 1, "Odds Ratio 95% CI ")
sheet1.write(0, 2, "P-value")


for i,col in enumerate(X.columns):
    X2 = sm.add_constant(X[col])
    
    #est = sm.OLS(y, X2)
    est = sm.Logit(y, X2)
    est2 = est.fit()    
    params = est2.params
    conf = est2.conf_int()
    conf['Odds Ratio'] = params
    conf.columns = ['5%', '95%', 'Odds Ratio']
    conf = np.exp(conf)
    conf['P-value'] = np.round(est2.pvalues.values,3)
    conf = conf.drop('const',axis=0)
    sheet1.write(i+1,0,str(col))
    sheet1.write(i+1,1,str("%0.2f (%0.2f - %0.2f)"%(conf['Odds Ratio'],conf['5%'],conf['95%'])))         
    sheet1.write(i+1,2,str("%0.3f "%(conf['P-value'])))         
    print((conf))
book.save(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results\univariate_analysis.xls") 
#df.to_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results\univariate_analysis.csv")

("Stroke"[MeSH Terms] OR "ischemic stroke"[Text Word] OR "cerebral infarct*"[Text Word] OR "brain infarct*"[Text Word]) 
AND "english"[Language] AND (("Artificial Intelligence"[MeSH Terms] OR "automat*"[Text Word] OR "algorithm*"[Text Word] OR
                              "machine learning"[Text Word] OR "deep learning"[Text Word] OR "convolutional neural network*"[Text Word])
                             AND "english"[Language]) AND (( "Alberta stroke program early CT score"[Text Word] 
                                                            OR "Alberta Stroke Program Early"[Text Word]))

#%% plotting correlation analysis
import seaborn as sns
import matplotlib.pyplot as plt

X = dt.correlation(X,thresh_corr,args) 

# Compute the correlation matrix
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(19, 17))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = True,fmt='.1g')

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
