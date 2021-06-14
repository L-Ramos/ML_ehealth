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
all_goal_phase = [4,5,6]
#all_goal_phase = [6]
min_goaldays = 6
time_hours = 72

for goal_phase in all_goal_phase:
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

#%% calibration plots

import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
     
#program = 'Alcohol'
#program = 'Cannabis'
program = 'Smoking'

path = os.path.join(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results",program+"72_safe_min_days_7_min_phase_6exp3\measures_RFC.pkl")

with open(path, 'rb') as f:
    meas = pickle.load(f)
    
probas = np.concatenate(meas.probas, axis=0 )    
y_test = np.concatenate(meas.labels, axis=0 )    

from sklearn.calibration import calibration_curve

fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, probas, n_bins=10)

fig = plt.figure(1, figsize=(10, 10))
            
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)


ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

ax1.plot(mean_predicted_value, fraction_of_positives, "s-")

path = os.path.join(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results",program+"72_safe_min_days_7_min_phase_6exp3\measures_LR.pkl")

with open(path, 'rb') as f:
    meas = pickle.load(f)
    
probas = np.concatenate(meas.probas, axis=0 )    
y_test = np.concatenate(meas.labels, axis=0 )    

from sklearn.calibration import calibration_curve

fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, probas, n_bins=10)


ax1.plot(mean_predicted_value, fraction_of_positives, "s-")


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, probas)
print(metrics.auc(fpr, tpr))

auc = list()
brier = list()
for y_test,probas in zip(meas.labels,meas.probas):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas)
    print(metrics.auc(fpr, tpr))
    auc.append(metrics.auc(fpr, tpr))
    brier.append(metrics.brier_score_loss(y_test, probas))
    
    
#%% calibrating a classifier   
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV


calibrated_clf = CalibratedClassifierCV(base_estimator=grid_lr.clf, cv=3)
calibrated_clf.fit(X_test,y_test)
probas = calibrated_clf.predict_proba(X_test)
print(metrics.brier_score_loss(y_test, probas[:,0]))
    
#%% YOuden index   
from methods import Mean_Confidence_Interval, find_optimal_cutoff


program = 'Alcohol'

path = os.path.join(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results",program+"72_safe_min_days_7_min_phase_6exp3\measures_RFC.pkl")

with open(path, 'rb') as f:
    meas = pickle.load(f)
    

sens = list()
spec = list()
ppv = list()
npv = list()
tn_l = list()
fp_l = list()
fn_l = list()
tp_l = list()


for y_test,probas in zip(meas.labels,meas.probas):
    t = find_optimal_cutoff(y_test,probas)[0]
    preds  = probas>=t
           
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, preds).ravel()  
    
    sens.append(tp/(tp+fn))
    spec.append(tn/(tn+fp))
    ppv.append(tp/(tp+fp))
    npv.append(tn/(tn+fn))
    
    tn_l.append(tn)
    fp_l.append(fp)
    fn_l.append(fn)
    tp_l.append(tp)
    
print(Mean_Confidence_Interval(sens))    
print(Mean_Confidence_Interval(spec))
print(Mean_Confidence_Interval(ppv))    
print(Mean_Confidence_Interval(npv))


array = [[np.sum(tp_l),np.sum(fp_l)],[np.sum(fn_l),np.sum(tn_l)]]
df_cm = pd.DataFrame(array, index = [i for i in ["Successful","Early Dropout"]],
                  columns = [i for i in ["Successful","Early Dropout"]])
plt.figure(figsize = (11,8))
sns.set(font_scale=2.0)
plot = sns.heatmap(df_cm, annot=True,fmt = ".1f",cmap="Blues")   
if program=='Smoking':
    program = 'Tobacco'
plot.set_title(str("Confusion matrix: "+program))

h,_ = os.path.split(path)
plot.figure.savefig(os.path.join(h,'confusion_matrix.png'))