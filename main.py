# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:08:45 2020

@author: laramos
"""


from sklearn.model_selection import StratifiedKFold

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
#time_hours_list = [72]

#time_hours_list = [48,96]
time_hours_list = [72]
#program_list = ['Alcohol','Cannabis', 'Smoking']
program_list = ['Smoking']
feature_type_list = ['safe']
experiment = 'exp3'
drop_afspraak = False
upthresh_corr = 0.8
lowthresh_corr = -10
#upthresh_corr = 0.99
#lowthresh_corr = -0.99
#min_goalphase = [4,5,6]
min_goalphase = [6]
min_goaldays = 7

max_date = '2020-10-12'

for goal_phase in min_goalphase:
    for program in program_list:
        for time_hours in time_hours_list:
            for feature_type in feature_type_list:
                print('')
            
                path_write = os.path.join(PATH_RESULTS,program+str(time_hours)+"_"+feature_type+"_min_days_"+str(min_goaldays)+"_min_phase"+str(goal_phase)+experiment)
        
                if not os.path.exists(path_write):
                    os.mkdir(path_write)
            
                df_temp = pd.read_csv(os.path.join(PATH_DATA,program+str(time_hours)+'.csv'))
                df_temp = df_temp[df_temp.StartDateOfParticipation<max_date]
                var_list = pd.read_csv(os.path.join(PATH_DATA,'var_list.csv'))
                
                args, feats_use  = dt.get_features(args, feature_type,df_temp.columns,var_list)
                
                X, y = dt.clean_data(df_temp,args,experiment,drop_afspraak,program,min_goaldays,goal_phase,feats_use)
                X, y = dt.fix_goal_variables(X,y,program)                
                
                dt.table_one(path_write,X,y,args,time_hours)  

                X,args = dt.correlation(X,upthresh_corr,lowthresh_corr,args)                 
                
                kf = StratifiedKFold(n_splits = args['outter_splits'], random_state=1, shuffle=True)
                
    
 
                
                rfc_m, lr_m, xgb_m, _, _ = ut.create_measures(args['outter_splits'])
                
                for fold,(train_index, test_index) in enumerate(kf.split(X,y)):
                    X_train,y_train = X.iloc[train_index,:], y[train_index]
                    X_test,y_test = X.iloc[test_index,:], y[test_index]
                    args['current_iteration'] = fold
                    args['pos_weights'] = (y_train.shape[0]-sum(y_train))/sum(y_train)
                    grid_rfc = mt.RandomForest_CLF(args,X_train,y_train,X_test,y_test,rfc_m,test_index)
                    #grid_xgb = mt.XGBoost_CLF(args,X_train,y_train,X_test,y_test,xgb_m,test_index)
                    grid_lr = mt.LogisticRegression_CLF(args,X_train,y_train,X_test,y_test,lr_m,test_index)
                    break
                break
                
                
                
                #names = ['RFC','LR','XGB']
                #meas = [rfc_m,lr_m,xgb_m]                            
                names = ['RFC','LR']
                meas = [rfc_m,lr_m]                            
                mt.print_results_excel(meas,names,path_write)
                mt.plot_shap(X,args,path_write, rfc_m.feat_names,var_list,names[0])
                #mt.plot_shap_xgb(X,args,path_write, xgb_m.feat_names,var_list,names[2])
                
                print("Auc: %0.2f, Sens: %0.2f, Spec: %0.2f, PPV: %0.2f, NPV: %0.2f"%(rfc_m.auc[1],rfc_m.sens[1],rfc_m.spec[1],rfc_m.ppv[1],rfc_m.npv[1]))
      
        

        
#%%plottinf confusion matrix for a given experiment
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
    

array = [[np.sum(meas.tp),np.sum(meas.fp)],[np.sum(meas.fn),np.sum(meas.tn)]]
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
book.save(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results\univariate_analysis_new.xls") 
#df.to_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results\univariate_analysis.csv")


#%% plotting correlation analysis
import seaborn as sns
import matplotlib.pyplot as plt

X = dt.correlation(X,thresh_corr,args) 

# Compute the correlation matrix
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(27, 25))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

#annot = True to show values oin the plot, but it is ugly
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .3},annot = False,fmt='f')



#%% Computing prevalence

np.random.seed(3)
list_idx = X.index.values
res = np.random.choice(list_idx,100)
y_prev = y[res]
print("Prevalence: ", sum(y_prev)/y_prev.shape[0])


#%%  plotting
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,10))

ax = sns.boxplot(data=X, orient="h", palette="Set2")

for var in X.columns:
    plt.plot(X[var])
    plt.show()
    
from scipy.stats import iqr
x = np.array([[10, 7, 4,3, 2, 1]])
iqr(x,rng=(25,75))

np.nanpercentile(x, 25),
np.nanpercentile(x, 75)
#%% getting the hyper-parameters chosen

import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
     
program = 'Alcohol'
#program = 'Cannabis'
#program = 'Smoking'

path = os.path.join(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results",program+"72_safe_min_days_6_min_phase_6exp3\measures_RFC.pkl")

with open(path, 'rb') as f:
    meas = pickle.load(f)
    
#var = 'clf__min_samples_split'
#var = 'clf__n_estimators'
#var = 'clf__min_samples_leaf'
var = 'clf__max_features'
#var = 'clf__max_depth'
#var = 'clf__criterion'
    
for vals in meas.best_params:
    print(vals[var])    


