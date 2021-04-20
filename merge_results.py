# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:11:34 2021

@author: laramos
"""

import os
import pandas as pd


PATH_RESULTS = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results"
#PATH_RESULTS = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results\Results_test_missing_goal\drop_particpants_mising_goal"
#PATH_RESULTS = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results"

program_list = ['Alcohol','Cannabis', 'Smoking']
#program_list = ['Alcohol']
feature_type = 'safe'
experiment = 'exp3'
drop_afspraak = False
thresh_corr = 0.8
goal_phase = 6
#min_goalphase = [6]
min_goaldays = 7
time_hours = 72

for i,program in enumerate(program_list):
    path_read = os.path.join(PATH_RESULTS,program+str(time_hours)+"_"+feature_type+"_min_days_"+str(min_goaldays)+"_min_phase_"+str(goal_phase)+experiment)
    df = pd.read_excel(os.path.join(path_read,'results.xls'))
    df['Program'] = program
    if i==0:
        df_merge = df
    else:
        df_merge = pd.concat([df_merge,df])
        
df_merge = df_merge[['Program','Methods', 'AUC 95% CI ', 'F1-Score', 'Sensitivity', 'Specificity','PPV', 'NPV']]
df_merge.to_excel(os.path.join(PATH_RESULTS,str(time_hours)+"_"+feature_type+"_min_days_"+str(min_goaldays)+"_min_phase_"+str(goal_phase)+experiment+'merged.xls'))


#%% this is for checking the Pr curve for the results

import pickle
from sklearn.metrics import accuracy_score,precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

path = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results\Final_manuscript\Alcohol72_safe_min_days_7_min_phase_6exp3\measures_RFC.pkl"



with open(path, 'rb') as f:        
    meas = pickle.load(f)

y = np.concatenate(meas.labels, axis=0 )
probas = np.concatenate(meas.probas, axis=0 )
    
precision, recall, _ = precision_recall_curve(y,probas)    

no_skill = len(y[y==1]) / len(y)

plt.figure(figsize=(10,7))
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic')

plt.title('PR curve - Smoking')
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

from sklearn.metrics import roc_auc_score
