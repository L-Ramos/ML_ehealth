# -*- coding: utf-8 -*-
"""
This code is responsible for reading the tables and converting them to usable features

Created on Wed Dec  9 12:25:16 2020

@author: laramos

For help contact me on: l.a.ramos.amc.@gmail.com

#Todo
#eventgenerators has missing ids, we can infer from ip if avaiable (is it accurate?)

"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

import utils_feature_engineering as uts

#path with the .csv files for the tables
PATH_DATA = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python"
#where to save the features created
PATH_SAVE = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\feature_datasets"
#We filter participants before this date, there was a major update in the intervention after that.
DATE_TO_FILTER = '2016-01-01'


#%% Loading Excel files

df_participator = uts.load_data(PATH_DATA,"Participator.xls") 

df_program = uts.load_data(PATH_DATA,"ProgramParticipation.xls") 

df_cons = uts.load_data(PATH_DATA,"ConsumptionRegistration.xls") 
df_day = uts.load_data(PATH_DATA,"ConsumptionDay.xls") 

df_phasestart = uts.load_data(PATH_DATA,"PhaseStart.xls") 

df_events = uts.load_data(PATH_DATA,"SystemEvent.txt") 

#AssignmentCompletion has information about ChallengeAssignment, they complete each other
df_assign = uts.load_data(PATH_DATA, "AssignmentCompletion.xls")
df_challenge = uts.load_data(PATH_DATA, "ChallengeAssignment.xls")

df_forumpost = uts.load_data(PATH_DATA,"ForumPost.xls")
df_forumthread = uts.load_data(PATH_DATA,"ForumThread.xls")

df_thread = uts.load_data(PATH_DATA,"ThreadViewing.xls")

df_pcusing = uts.load_data(PATH_DATA, "AssignmentAnswerProsConsUsing.csv")

df_assign_answer = uts.load_data(PATH_DATA, "AssignmentAnswerAgreements.csv")

df_partbadge = uts.load_data(PATH_DATA,"ParticipationBadge.xls")

df_achievelike = uts.load_data(PATH_DATA,"AchievementLike.xls")

df_diary = uts.load_data(PATH_DATA,"DiaryRecord.csv")

# fix some column type issues with dataframes
df_program, df_participator, df_phasestart, df_events, df_forumpost, df_forumthread, df_thread = uts.fix_frames(df_program, df_participator, df_phasestart, df_events, df_forumpost,
                                                                                     df_forumthread, df_thread)

print("Data Loaded")

df_events = df_events.dropna(subset=['EventGenerator'])


#%% Filtering by date, so we include only after a certain year

df_participator = uts.filter_date(df_participator,'DateCreated',DATE_TO_FILTER)
df_program = uts.filter_date(df_program,'StartDateOfParticipation',DATE_TO_FILTER)
df_cons = uts.filter_date(df_cons,'DateOfRegistration',DATE_TO_FILTER)
df_phasestart = uts.filter_date(df_phasestart,'DateStarted',DATE_TO_FILTER)
df_events = uts.filter_date(df_events,'DateOfEvent',DATE_TO_FILTER)
df_forumpost = uts.filter_date(df_forumpost,'DateCreated',DATE_TO_FILTER)
df_forumthread = uts.filter_date(df_forumthread,'DateCreated',DATE_TO_FILTER)
df_thread = uts.filter_date(df_thread, 'Date',DATE_TO_FILTER)
df_diary = uts.filter_date(df_diary, 'DateCreated',DATE_TO_FILTER)

print('Tables filtered by Date.')


#%% Merge both ProgramParticipation with the rest
# many tables connect to ProgramParticipation and the information there is needed for furthe processing

#Table to be merged with all the others, it has the main participant information
df_merge = df_program

df_merge_part = pd.merge(df_participator,df_merge,left_on='Id',right_on='Participator')
print(df_participator.shape[0],df_merge_part.shape[0])

df_merge_cons = pd.merge(df_cons,df_merge,left_on='Participation',right_on='Id')
print(df_cons.shape[0],df_merge_cons.shape[0])

df_merge_assign = pd.merge(df_assign,df_merge,left_on='Participation',right_on='Id',how='inner')
print(df_merge_assign.shape[0],df_assign.shape[0])

df_merge_challenge = pd.merge(df_assign,df_challenge,left_on='Assignment',right_on='Id',how='inner')
print(df_assign.shape[0],df_merge_challenge.shape[0])

df_merge_forumpost = uts.merge_multactivity_program(df_forumpost,df_program,'DateCreated','StartDateOfParticipation','Author','Participator')
print(df_forumpost.shape[0],df_merge_forumpost.shape[0])

df_merge_forumthread = uts.merge_multactivity_program(df_forumthread,df_program,'DateCreated','StartDateOfParticipation','Author','Participator')
print(df_forumthread.shape[0],df_merge_forumthread.shape[0])

print("Tables merged.")
#%% Feature engineering per program type and for different time frames

#dictionaries for converting the data from numbers to text
program_dict = {1: 'Alcohol',2: 'Cannabis', 5: 'Smoking'}
pros_cons_dic = {3:'Pros Short', 4: 'Pros Long', 5: 'Cons Short', 6: 'Cons Long'} 

#How many hours of data to use for generating the features, one file per program and per time will be generated
TIME_HOURS_LIST = [48,72,96,120]

NUMBER_DAYS_CONS = 7

#Main for loop, controls programs and time intervals
for PROGRAM in program_dict.keys():
    for TIME_HOURS in TIME_HOURS_LIST:
 
        # First get participants of program type challenge and only the assignments from phase 1
        # df_filter has all the participants that should be included in the feature engineering
        df_merge_challenge_feat, df_filter = uts.select_participants_from_phase(df_program, PROGRAM, df_merge_challenge, df_phasestart)
               
        # filtering by time and adding df_filter columns to df_merge_challenge
        df_merge_challenge_feat = uts.filter_time_tables(TIME_HOURS, df_filter, df_merge_challenge_feat,'assignment','DateCompletedUtc')
        
        # First step of feature engineering, starts with pros and cons
        df_merge_feats = uts.feature_engineering_pros_cons(df_filter,df_pcusing,pros_cons_dic)
                            
        # Feature engineering for binary features
        df_merge_feats = uts.feature_engineering_assignments(df_merge_challenge_feat,df_filter,df_program,df_merge_feats)
                
        # Feature engineering for time from start to assignment completion - confusing so not included anymore
        #df_merge_feats = uts.feature_engineering_time_assigment(df_merge_challenge_feat,df_merge_feats,df_program,'DateRegistered','DateCompletedUtc')
            
        # Feature Engineering for AssignmentAnswerAgreements (length of reply / number of replies)
        df_merge_feats = uts.feature_engineering_assignment_agreement(df_filter,df_merge_feats,df_assign_answer,pros_cons_dic)
        
        # Feature engineering from consumption target and goal
        df_merge_feats = uts.feature_engineering_consumption_features(df_day,df_merge_feats)
           
        # Feature engineering for Forum and Login Events
        df_merge_feats = uts.feature_engineering_forum_and_login_events(TIME_HOURS,df_filter,df_merge_feats,df_events)
        
        # Forum Post Feature engineering
        df_merge_feats = uts.feature_engineering_forumpost_thread(TIME_HOURS,df_filter,df_forumpost,df_merge_feats,'DateCreated','Number of ForumPosts')
        
        # Forum Thread Feature engineering
        df_merge_feats = uts.feature_engineering_forumpost_thread(TIME_HOURS,df_filter,df_forumthread,df_merge_feats,'DateCreated','Number of ThreadPosts')
        
        #removing the particpants that access assignement multiple times
        #df_merge_feats = uts.remove_user_future_edits(df_merge_challenge,df_filter,df_merge_feats,TIME_HOURS)
        
        # Fetch the consumption in units before the start of the last phase (NUMBER_DAYS_CONS controls how many days are returned - days of the week)
        df_merge_feats = uts.add_consumption_before_last_phase(NUMBER_DAYS_CONS,df_cons,df_phasestart, df_merge_feats)
             
        # Feature engineering about the units consumed in the first days of intervetion use        
        df_merge_feats = uts.feature_engineering_consumption_first_days(TIME_HOURS, df_cons, df_filter,df_merge_feats)
        
        # Feature engineering for the total number of forum threads viewed by the user
        df_merge_feats = uts.feature_engineering_thread_vieweing(TIME_HOURS,df_thread,df_filter,df_merge_feats)
        
        # Feature engineering for the total number of badges earned by the user
        df_merge_feats = uts.feature_engineering_badge(TIME_HOURS,df_filter,df_partbadge,df_merge_feats)
        
        # Feature engineering for the total number of achievements liked/viewed by the user
        df_merge_feats = uts.feature_engineering_achievementlike(TIME_HOURS,df_filter,df_achievelike,df_merge_feats)
        
        # Feature engineering for the total number of diary entries the user did
        df_merge_feats = uts.feature_engineering_diaryrecord(TIME_HOURS,df_filter,df_diary,df_merge_feats)
        
        #Saves the dataframe with the features engineered as a .csv file
        uts.save_frame(df_merge_feats,os.path.join(PATH_SAVE,program_dict[PROGRAM]+str(TIME_HOURS)+'.csv'))
        print("saved")


# df_temp = pd.merge(df,df_phase_filter_temp, on='Participation',how='left')

# df = df_temp[df_temp.DateStarted.notna()]


#%% #this is to organize the consumption data 
import matplotlib.pyplot as plt

program_dict = {1: 'Alcohol',2: 'Cannabis', 5: 'Smoking'}

max_pause = [1,2,3,4,5]

df = df_merge_cons
df = df[df.Program==1]

df = df.sort_values(by=['Participation','DateOfRegistration'])

df_phase6 = df_phasestart[df_phasestart.Phase==6]

df = df[df['Phase_y']==6]

df_merge = pd.merge(df,df_phase6,on='Participation')


df_merge = df_merge[['Participation','DateOfRegistration','NumberOfUnitsConsumed','Phase_y','StartDateOfParticipation','DateStarted']]
df_merge['DateOfRegistration'] = pd.to_datetime(df_merge['DateOfRegistration'])
df_merge['DateStarted'] = pd.to_datetime(df_merge['DateStarted'])

df_after_end = df_merge[df_merge['DateOfRegistration']>df_merge['DateStarted']]

pat_list = df_after_end.Participation.unique()

list_plots = list()

for e in max_pause:

    df = df_after_end[df_after_end.Participation==pat_list[0]]
    df['diff'] = df['DateOfRegistration'].diff()
    df = df.reset_index(drop=True)
    df_final = df[:df[df['diff'].dt.days>e].index[0]]

    for i in range(1,len(pat_list)):
        pat = pat_list[i]             
        df = df_after_end[df_after_end.Participation==pat]
        df['diff'] = df['DateOfRegistration'].diff()
        df = df.reset_index(drop=True)
        if any(df['diff'].dt.days>e)==True:
            df_loop = df[:df[df['diff'].dt.days>e].index[0]]
        else:
            df_loop = df
        df_final = pd.concat([df_final,df_loop])
        
    df_unique = df_final['Participation'].value_counts()
    
    hist_plot = list()
    for i in range(1,100):
        hist_plot.append(sum(df_unique>i))
    list_plots.append(hist_plot)

plt.figure(figsize=(12,7))
plt.plot(list_plots[0],'r',label='Interval days = 1')
plt.plot(list_plots[1],'g',label='Interval days = 2')
plt.plot(list_plots[2],'b',label='Interval days = 3')
plt.plot(list_plots[3],'y',label='Interval days = 4')
plt.plot(list_plots[4],'k',label='Interval days = 5')
#plt.hist(df_unique,bins=20,range=[0,100])
    

plt.xlabel('Days')
plt.ylabel('Total Participants')
#plt.ylim((0,250))
plt.title(str('Continuous consumption registration' ))
plt.legend(loc='upper right')
s = np.arange(0, 105, 5)
s[0] = 1
plt.xticks(s)
plt.show()



#%% Plotting some events from a participant using only the events table
remove_after_phase1 = False

if remove_after_phase1:
    df_merge_challenge_feat,df_filter = uts.filter_assigment_ids_after_phase(['Voor- en nadelen','Jouw afspraken'],df_merge_challenge_feat,
                                                                             df_phasestart,df_filter)
        
        
counts = df_events.EventGenerator.value_counts()


#ids = 29185
#ids = 39708
#ids = 4237
ids = 11804

df = df_events[df_events.EventGenerator==ids]

df['DateOfEvent'] = pd.to_datetime(df['DateOfEvent'])
df['Date'] = df['DateOfEvent'].dt.date

vals_events = df['Date'].value_counts()
vals_login = df[df['Title']=='User Login']['Date'].value_counts()
vals_forum = df[df['Title']=='Forum Visited']['Date'].value_counts()

df_plot_events = pd.DataFrame(columns=['Date','Events'])
df_plot_events['Date'] = (vals_events.index)
df_plot_events['Events'] = np.array(vals_events.values)

df_plot_login = pd.DataFrame(columns=['Date','Login'])
df_plot_login['Date'] = (vals_login.index)
df_plot_login['Login'] = np.array(vals_login.values)

df_plot_forum = pd.DataFrame(columns=['Date','Forum'])
df_plot_forum['Date'] = (vals_forum.index)
df_plot_forum['Forum'] = np.array(vals_forum.values)

df_plot_el = pd.merge(df_plot_events,df_plot_login,on='Date',how='left')
df_plot = pd.merge(df_plot_el,df_plot_forum,on='Date',how='left')





#%% This is for plotting the number of participants over the time

import os
import matplotlib.pyplot as plt

def plot_figure(res):
        list_vals = list()
    for i in range(1,10):
        y = (res>=i)
        list_vals.append(sum(y))
        print(sum(y))
    
    #list_vals.append(0)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    
    ax.set_ylim(0, max(list_vals))
     
    ax.set_title(name)
    ax.set_xlabel("Days")
    ax.set_ylabel("Participants")  


#name = "Alcohol"
#name = "Cannabis"
name = "Smoking"
df = pd.read_csv(os.path.join(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\feature_datasets",name+"72.csv"))

df1 = df_merge_part[['Id_y','Participator']]

df_merge = pd.merge(df,df1,left_on='Id',right_on='Id_y')

df_merge = df_merge.rename(columns = {'Id_y': 'ID'})

#df_events['EventGenerator'] = pd.to_numeric(df_events['EventGenerator'], errors='coerce').astype(pd.Int64Dtype())
#df_merge['Participator'] = pd.to_numeric(df_merge['Participator'], errors='coerce').astype(pd.Int64Dtype())

df_events['DateOfEvent'] = pd.to_datetime(df_events['DateOfEvent'])
df_merge['StartDateOfParticipation'] = pd.to_datetime(df_merge['StartDateOfParticipation'])


df_merge2 = uts.merge_multactivity_program(df_events,df_merge, 'DateOfEvent','StartDateOfParticipation','EventGenerator','Participator')

df_mergef = pd.merge(df_merge2,df,left_on='ID',right_on='Id',how='inner')

res = df_mergef.groupby('ID')['DateOfEvent'].nunique()

plot_figure(res)

#%%  this is for plotting ghe registry of drinking over time

import os

def plot_comsumption(res,days):
    list_vals = list()
    for i in range(1,days):
        y = (res>=i)
        list_vals.append(sum(y))
        print(sum(y))
    
    #list_vals.append(0)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    
    ax.set_ylim(0, max(list_vals))
    ax.set_xlim(1, days)
    ax.xaxis.set_ticks(np.arange(1, days-1, 1))
    
    ax.set_title(name)
    ax.set_xlabel("Days")
    ax.set_ylabel("Participants") 
    ax.plot(list_vals)

name = "Alcohol"
#name = "Cannabis"
#name = "Smoking"
df = pd.read_csv(os.path.join(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\feature_datasets",name+"72.csv"))

df1 = df_cons[['Participation','DateOfRegistration','NumberOfUnitsConsumed']]

df_merge = pd.merge(df,df1,left_on='Id',right_on='Participation')

df_merge['StartDateOfParticipation'] = pd.to_datetime(df_merge['StartDateOfParticipation'])
df_merge['DateOfRegistration'] = pd.to_datetime(df_merge['DateOfRegistration'])

res = df_merge.groupby('Id')['DateOfRegistration'].nunique()

plot_comsumption(res,30)

#%%
# difference for the consecutive days
df_merge = df_merge.sort_values('DateOfRegistration', ascending=0)

df_merge['diff_col'] = df_merge.sort_values(['Id','DateOfRegistration','StartDateOfParticipation']).groupby('Id')['DateOfRegistration'].diff()
df_merge = df_merge[['Id','StartDateOfParticipation','DateOfRegistration','NumberOfUnitsConsumed','diff_col']]

#remove days that have alrge intervals
df2 = df_merge[df_merge['diff_col'].dt.days<=1]


res = df2.groupby('Id')['DateOfRegistration'].nunique()

plot_comsumption(res,30)


df3 = df_merge[df_merge.diff_col.isna()]
df3['diff_col'] = df3['DateOfRegistration'] - df3['StartDateOfParticipation']

df3 = df3[df3['diff_col'].dt.days>1]

df4 = df_merge[df_merge['Id']== 14763]

df5 = df_merge.head(1000)
