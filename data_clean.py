# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:17:31 2021

@author: laramos
"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import xlwt
import numpy as np
import os


def clean_data(df_temp,args,experiment,drop_afspraak, feats_use=list()):
    
    goal_dict = {0: 'Missing', 1: 'Stop', 2: 'Reduce', 3: 'Slowly Stop', 4: 'Slowly Reduce'}
     
    if len(feats_use)==0:
        feats_use = df_temp.columns
   
    if experiment=='exp1':
        X = df_temp
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
    
    X['GoalOfProgram'] = X['GoalOfProgram'].map(goal_dict) 
       
    scaler = ColumnTransformer([('Program Goal', OneHotEncoder(),args['mask_cat'])], remainder='passthrough')
    #scaler = ColumnTransformer([(goal_dict.values(), OneHotEncoder(),args['mask_cat'])], remainder='passthrough')
    X = scaler.fit_transform(X)   
    X = pd.DataFrame(X,columns = scaler.get_feature_names())
    
    if drop_afspraak:
        cols_todrop = [c for c in X.columns if 'Afsprak' in c or 'afsprak' in c]
        X = X.drop(cols_todrop,axis=1)
        args['mask_cont'] = [c for c in args['mask_cont'] if 'Afsprak' not in c and 'afsprak' not in c]
    return(X,y)


def get_features(args,feature_type,feature_names,var_list):

    if feature_type == 'safe':
            var_list = var_list[var_list['Safe']==1]
            
    args['mask_cat'] = list(var_list[var_list['Type']=='cat'].Feature)
    args['mask_bin'] = list(var_list[var_list['Type']=='bin'].Feature)
    args['mask_cont'] = list(var_list[var_list['Type']=='cont'].Feature)
    feats_use = list(var_list.Feature)       
            
    return(args,feats_use)


def table_one(path_write,X,y,args,time_hours):

    df = X.copy()
    df['Phase'] = y
    
    book = xlwt.Workbook(encoding="utf-8")    
    sheet1 = book.add_sheet("Sheet 1")
    
    #n_part = df.shape[0]
    df_phase1 = df[df.Phase==0]
    n_phase1 = df_phase1.shape[0]
    
    df_other = df[df.Phase==1]
    n_other = df_other.shape[0]

    cont_cols = args['mask_cont']
    cat_cols = args['mask_cat']
    bin_cols = args['mask_bin']

    
    for i,col in enumerate(cont_cols):    
        sheet1.write(i+1,0,col)    

    for col in (bin_cols):
        i = i+1
        sheet1.write(i+1,0,col)  
    
    for col in (cat_cols):
        i = i+1
        sheet1.write(i+1,0,col)      
        
    sheet1.write(0,1,"Alcohol Phase 1")         
    sheet1.write(0,2,"Alcohol Other Phases")
        
    for i,col in enumerate(cont_cols):
        sheet1.write(i+1,1,str("%0.0f (%0.2f)"%(np.mean(df_phase1[col]),np.std(df_phase1[col]))))         
        sheet1.write(i+1,2,str("%0.0f (%0.2f)"%(np.mean(df_other[col]),np.std(df_other[col]))))
        
    for col in (bin_cols):
        i = i + 1
        sheet1.write(i+1,1,str("%0.0f (%0.2f)"%(np.sum(df_phase1[col]),((np.sum(df_phase1[col])*100)/n_phase1))))
        sheet1.write(i+1,2,str("%0.0f (%0.2f)"%(np.sum(df_other[col]),((np.sum(df_other[col])*100)/n_other))))
        #sheet1.write(i+1,2,str("%0.0f (%0.2f)"%(np.mean(df_other[col]),np.std(df_other[col]))))  
    i = i+1
    cat_cols = [c for c in df.columns if 'Program Goal' in c]
    for col in cat_cols:
        col_print = col.replace('__x0_',': ')
        i = i + 1
        sheet1.write(i+1,0,col_print) 
        sheet1.write(i+1,1,str("%0.0f (%0.2f)"%(np.sum(df_phase1[col]),((np.sum(df_phase1[col])*100)/n_phase1))))
        sheet1.write(i+1,2,str("%0.0f (%0.2f)"%(np.sum(df_other[col]),((np.sum(df_other[col])*100)/n_other))))
        #sheet1.write(i+1,2,str("%0.0f (%0.2f)"%(np.mean(df_other[col]),np.std(df_other[col]))))  
        
    sheet1.write(i+2,0,str("Number per class"))        
    sheet1.write(i+2,1,str("%0.0f "%(df_phase1.shape[0])))
    sheet1.write(i+2,2,str("%0.0f "%(df_other.shape[0])))
    book.save(os.path.join(path_write,"_Descriptive.xls")) 
