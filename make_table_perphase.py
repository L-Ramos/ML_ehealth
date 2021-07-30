# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:33:38 2021

@author: laramos
"""
import pandas as pd


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

max_date = '2020-10-12'
df = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\feature_datasets\Cannabis72.csv")
df = df[df.StartDateOfParticipation<max_date]
tot = df.shape[0]
df = get_success_label(df)

df = df[~df['GoalOfProgram'].isna()]

#drop out per fase
for phase in range(0,7):
    print(df[df.Phase==phase].shape[0],(df[df.Phase==phase].shape[0]*100)/tot)

#decrease reach per fase
for phase in range(1,7):
    print(df[df.Phase>=phase].shape[0],(df[df.Phase>=phase].shape[0]*100)/tot)
    
    
print("\n")
for phase in range(0,8):    
    print(df[df['Total_achieved']>=phase].shape[0],(df[df['Total_achieved']>=phase].shape[0]*100)/tot)
    


df = df[df.Phase>=6]    

print(df[df['Total_achieved']<7].shape[0],(df[df['Total_achieved']>=phase].shape[0]*100)/tot)

