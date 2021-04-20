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

def correlation(dataset, upthreshold,lowthreshold,args):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            #if (corr_matrix.iloc[i, j] >= upthreshold) and (corr_matrix.columns[j] not in col_corr):
            if ((corr_matrix.iloc[i, j] >= upthreshold) or (corr_matrix.iloc[i, j] <= lowthreshold)) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    new_mask_cont = [var for var in args['mask_cont'] if var in dataset.columns]              
    new_mask_cat = [var for var in args['mask_cat'] if var in dataset.columns]
    new_mask_bin = [var for var in args['mask_bin'] if var in dataset.columns]    
    args['mask_cont'] = new_mask_cont
    args['mask_cat'] = new_mask_cat
    args['mask_bin'] = new_mask_bin

    return(dataset,args)

def fix_goal_variables(X,y,program):     
    if program=='Alcohol':
        X['y'] = y
        X = X[X['Program Goal-Missing']!=1]
        X = X.reset_index(drop=True)
        y = X['y']
        X = X.drop(['y'],axis=1)
        X = X.drop(['Program Goal-Missing'],axis=1)
    elif program=='Smoking':
        X['y'] = y
        X = X[X['Program Goal-Reduce']!=1]
        X = X.reset_index(drop=True)
        y = X['y']
        X = X.drop(['y'],axis=1)
        X = X.drop(['Program Goal-Reduce'],axis=1)
    return(X,y)

def get_success_label(df_temp):

    list_goal = ['GoalMonday', 'GoalTuesday', 'GoalWednesday', 'GoalThursday', 'GoalFriday', 'GoalSaturday', 'GoalSunday']
    list_target = ['MondayTarget', 'TuesdayTarget', 'WednesdayTarget', 'ThursdayTarget', 'FridayTarget', 'SaturdayTarget', 'SundayTarget']
    list_cons = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    
    df_temp['GoalMonday'] = df_temp['Monday']<=df_temp['MondayTarget']
    df_temp['GoalTuesday'] = df_temp['Tuesday']<=df_temp['TuesdayTarget']
    df_temp['GoalWednesday'] = df_temp['Wednesday']<=df_temp['WednesdayTarget']
    df_temp['GoalThursday'] = df_temp['Thursday']<=df_temp['ThursdayTarget']
    df_temp['GoalFriday'] = df_temp['Friday']<=df_temp['FridayTarget']
    df_temp['GoalSaturday'] = df_temp['Saturday']<=df_temp['SaturdayTarget']
    df_temp['GoalSunday'] = df_temp['Sunday']<=df_temp['SundayTarget']
    
    df_temp['Total_achieved'] = df_temp[list_cons].apply(lambda x: x.count(), axis=1)
    return(df_temp)

def clean_data(df_temp,args,experiment,drop_afspraak,program, total_goaldays = 0,min_goalphase = 0, feats_use=list()):
    
    goal_dict = {0: 'Missing', 1: 'Stop', 2: 'Reduce', 3: 'Slowly Stop', 4: 'Slowly Reduce'}
     
    if len(feats_use)==0:
        feats_use = df_temp.columns
   
    if experiment=='exp1':
        X = df_temp
        y = (df_temp['Phase']>1).astype('int32')
    elif experiment=='exp2':
        X1 = df_temp[df_temp['Phase']==2]
        X2 = df_temp[df_temp['Phase']==6]
        frames = [X1,X2]
        X = pd.concat(frames)
        y = (X['Phase']>2).astype('int32').reset_index(drop=True)
    elif experiment=='exp3':
        get_success_label(df_temp)
        X1 = df_temp[df_temp['Phase']==2]
        X2 = df_temp[df_temp['Phase']>=min_goalphase]        
        frames = [X1,X2]
        X = pd.concat(frames)
        y = ((X['Phase']>2) & (X['Total_achieved']>=total_goaldays)).astype('int32').reset_index(drop=True)
        
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
    cols = scaler.get_feature_names()
    
    for i,c in enumerate(cols):
        cols[i] = c.replace('__x0_','-')
        
    X = pd.DataFrame(X,columns = cols)
    
    if drop_afspraak:
        cols_todrop = [c for c in X.columns if 'Afsprak' in c or 'afsprak' in c]
        X = X.drop(cols_todrop,axis=1)
        args['mask_cont'] = [c for c in args['mask_cont'] if 'Afsprak' not in c and 'afsprak' not in c]
    if program == 'Smoking':
        cols_todrop = [c for c in X.columns if 'Target' in c]
        X = X.drop(cols_todrop,axis=1)
        args['mask_cont'] = [c for c in args['mask_cont'] if 'Target' not in c]
        
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
        sheet1.write(i+1,1,str("%0.0f (%0.0f - %0.0f)"%(np.median(df_phase1[col]),np.nanpercentile(df_phase1[col], 25,interpolation = 'midpoint'),np.nanpercentile(df_phase1[col], 75,interpolation = 'midpoint'))))         
        sheet1.write(i+1,2,str("%0.0f (%0.0f - %0.0f)"%(np.median(df_other[col]),np.nanpercentile(df_other[col], 25,interpolation = 'midpoint'),np.nanpercentile(df_other[col], 75,interpolation = 'midpoint'))))
        
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
