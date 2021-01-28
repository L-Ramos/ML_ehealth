# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:17:31 2021

@author: laramos
"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

def clean_data(df_temp,args,experiment, feats_use=list()):
    
    goal_dict = {0: 'Missing', 1: 'Stop', 2: 'Reduce', 3: 'Slowly Stop', 4: 'Slowly Reduce'}
     
    if len(feats_use)==0:
        feats_use = df_temp.columns
   
    if experiment=='exp1':
        y = (df_temp['Phase']>1).astype('int32')
    else:
        X1 = df_temp[df_temp['Phase']==2]
        X2 = df_temp[df_temp['Phase']==6]
        frames = [X1,X2]
        X = pd.concat(frames)
        y = (X['Phase']>2).astype('int32').reset_index(drop=True)
        
    X = X[feats_use]  
    #X['Jouw afspraken'] = X['Jouw afspraken'].fillna(0)
    #X['Afspraken maken'] = X['Afspraken maken'].fillna(0)
    #X['Start video'] = X['Start video'].fillna(0)
    #X['Voor- en nadelen'] = X['Voor- en nadelen'].fillna(0)
    X = X.fillna(0)
    
    df_temp = df_temp.replace('GoalOfProgram', goal_dict)
    
    
    for key in goal_dict.keys():
        X['GoalOfProgram'] = X
    
    scaler = ColumnTransformer([('Program Goal', OneHotEncoder(),args['mask_cat'])], remainder='passthrough')
    X = scaler.fit_transform(X)   
    X = pd.DataFrame(X,columns = scaler.get_feature_names())
    
    return(X,y)

# def get_features(args,feature_type,feature_names):

#     init_feat = [c for c in feature_names if 'Initial' in c]
#     targ_feat = [c for c in feature_names if 'Target' in c]
#     feats = init_feat + targ_feat
#     all_feats = ['GoalOfProgram','Pros Short', 'Pros Long', 'Cons Short', 'Cons Long', 'Start video','Voor- en nadelen','Afspraken maken', 'Jouw afspraken','Number of Agreements', 'Total Agreement Length']
#     all_feats = all_feats + feats
#     if feature_type == 'all':
#         args['mask_cat'] = ['GoalOfProgram']
#         args['mask_cont'] = ['Pros Short', 'Pros Long', 'Cons Short', 'Cons Long','Number of Agreements', 'Total Agreement Length'] + feats
#         feats_use = all_feats
#     else:
#         if feature_type == 'safe':
#            args['mask_cat'] = ['GoalOfProgram']
#            args['mask_cont'] = feats
#            feats_use =  ['GoalOfProgram','Start video','Voor- en nadelen','Afspraken maken', 'Jouw afspraken'] + feats
#     return(args,feats_use)

def get_features(args,feature_type,feature_names,var_list):

    if feature_type == 'all':        
        args['mask_cat'] = list(var_list[var_list['Type']=='cat'].Feature)
        args['mask_cont'] = list(var_list[var_list['Type']=='cont'].Feature)
        feats_use = list(var_list.Feature)
    else:
        if feature_type == 'safe':
            var_list = var_list[var_list['Safe']==1]
            args['mask_cat'] = list(var_list[var_list['Type']=='cat'].Feature)
            args['mask_cont'] = list(var_list[var_list['Type']=='cont'].Feature)
            feats_use = list(var_list.Feature)
    return(args,feats_use)