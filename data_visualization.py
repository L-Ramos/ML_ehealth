# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:25:16 2020

@author: laramos

To Do:
SystemEvents  
eventgenerators are missing, we can infer from ip if avaiable

"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import utils_data_visualization as uts


PATH_DATA = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python"
PATH_SAVE = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\feature_datasets"


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

#events did not work, no matchs with the data
#df_partevents = uts.load_data(PATH_DATA,"ParticipationEvents.xls")

df_achievelike = uts.load_data(PATH_DATA,"AchievementLike.xls")

df_program, df_participator, df_phasestart, df_events, df_forumpost, df_forumthread, df_thread = uts.fix_frames(df_program, df_participator, df_phasestart, df_events, df_forumpost,
                                                                                     df_forumthread, df_thread)

print("Data Loaded")

#FIX THIS
#eventgenerators are missing, we can infer from ip if avaiable
df_events = df_events.dropna(subset=['EventGenerator'])
#print("Unique Ids in SystemEvent: ",df_events.EventGenerator.nunique())


#%% Filtering by date, so we include only after a certain year

df_participator = uts.filter_date(df_participator,'DateCreated')
df_program = uts.filter_date(df_program,'StartDateOfParticipation')
df_cons = uts.filter_date(df_cons,'DateOfRegistration')
df_phasestart = uts.filter_date(df_phasestart,'DateStarted')
df_events = uts.filter_date(df_events,'DateOfEvent')
df_forumpost = uts.filter_date(df_forumpost,'DateCreated')
df_forumthread = uts.filter_date(df_forumthread,'DateCreated')
df_thread = uts.filter_date(df_thread, 'Date')

print('Tables filtered by Date.')


#%% Merge both ProgramParticipation with the rest

df_merge = df_program

df_merge_part = pd.merge(df_participator,df_merge,left_on='Id',right_on='Participator')
print(df_participator.shape[0],df_merge_part.shape[0])

df_merge_cons = pd.merge(df_cons,df_merge,left_on='Participation',right_on='Id')
print(df_cons.shape[0],df_merge_cons.shape[0])

# df_merge_events = pd.merge(df_events,df_merge,left_on='EventGenerator',right_on='Id',how='inner')
# print(df_events.shape[0],df_merge_events.shape[0])

df_merge_assign = pd.merge(df_assign,df_merge,left_on='Participation',right_on='Id',how='inner')
print(df_merge_assign.shape[0],df_assign.shape[0])

df_merge_challenge = pd.merge(df_assign,df_challenge,left_on='Assignment',right_on='Id',how='inner')
print(df_assign.shape[0],df_merge_challenge.shape[0])

df_merge_forumpost = uts.merge_multactivity_program(df_forumpost,df_program,'DateCreated','StartDateOfParticipation','Author','Participator')
print(df_forumpost.shape[0],df_merge_forumpost.shape[0])

df_merge_forumthread = uts.merge_multactivity_program(df_forumthread,df_program,'DateCreated','StartDateOfParticipation','Author','Participator')
print(df_forumthread.shape[0],df_merge_forumthread.shape[0])

#df_merge_risk = pd.merge(df_assign,df_assign_risk,left_on='Assignment',right_on='Id',how='inner')
print("Tables merged.")
#%% Selecting assigments only from Participants from PROGRAM Phase 1 and > 2


program_dict = {1: 'Alcohol',2: 'Cannabis', 5: 'Smoking'}
pros_cons_dic = {3:'Pros Short', 4: 'Pros Long', 5: 'Cons Short', 6: 'Cons Long'} 

TIME_HOURS_LIST = [72,96,120]

number_of_days_cons = 7

for PROGRAM in program_dict.keys():
    for TIME_HOURS in TIME_HOURS_LIST:
 
        df_merge_challenge_feat, df_filter = uts.select_participants_from_phase(df_program, PROGRAM, df_merge_challenge, df_phasestart)
               
        #filtering by time
        df_merge_challenge_feat = uts.filter_time_tables(TIME_HOURS, df_filter, df_merge_challenge_feat,'assignment','DateCompletedUtc')
        
        #Feature engineering for pros and cons
        df_merge_feats = uts.feature_engineering_pros_cons(df_filter,df_pcusing,pros_cons_dic)
            
        #Feature engineering for binary features
        df_merge_feats = uts.feature_engineering_binary(df_merge_challenge_feat,df_filter,df_program,df_merge_feats)
        
        #Feature engineering for time from start to assignment completion
        #df_merge_feats = uts.feature_engineering_time_assigment(df_merge_challenge_feat,df_merge_feats,df_program,'DateRegistered','DateCompletedUtc')
            
        #Feature Engineering for AssignmentAnswerAgreements (length of reply / number of replies)
        df_merge_feats = uts.feature_engineering_assignment_agreement(df_filter,df_merge_feats,df_assign_answer,pros_cons_dic)
        
        # Feature engineering from consumption target and goal
        df_merge_feats = uts.feature_engineering_consumption_features(df_day,df_merge_feats)
           
        #Feature engineering for Forum and Login Events
        df_merge_feats = uts.feature_engineering_forum_and_login_events(TIME_HOURS,df_filter,df_merge_feats,df_events)
        
        #Forum Post FEature engineering
        df_merge_feats = uts.feature_engineering_forumpost(TIME_HOURS,df_filter,df_forumpost,df_merge_feats,'DateCreated','Number of ForumPosts')
        
        df_merge_feats = uts.feature_engineering_forumpost(TIME_HOURS,df_filter,df_forumthread,df_merge_feats,'DateCreated','Number of ThreadPosts')
        
        #removing the particpants that access assignement multiple times
        #df_merge_feats = uts.remove_ids_future_edits(df_merge_challenge,df_filter,df_merge_feats,TIME_HOURS)
        
        df_merge_feats = uts.add_consumption_before_last_phase(number_of_days_cons,df_cons,df_phasestart, df_merge_feats)
        
        df_merge_feats = uts.feature_engineering_consumption_first_days(time_hours, df_cons, df_filter,df_merge_feats)
        
        df_merge_feats = uts.feature_engineering_thread_vieweing(TIME_HOURS,df_thread,df_filter,df_merge_feats)
        
        df_merge_feats = uts.feature_engineering_badge(TIME_HOURS,df_filter,df_partbadge,df_merge_feats)
        
        df_merge_feats = uts.feature_engineering_achievementlike(TIME_HOURS,df_filter,df_achievelike,df_merge_feats)
        
        #Saves it as a .csv file
        uts.save_frame(df_merge_feats,os.path.join(PATH_SAVE,program_dict[PROGRAM]+str(TIME_HOURS)+'.csv'))
        print("saved")


# df_temp = pd.merge(df,df_phase_filter_temp, on='Participation',how='left')

# df = df_temp[df_temp.DateStarted.notna()]


#%%

print((203*100)/9709)


#%%This is for df_achievelike

df_initial = uts.compute_target_consumption_features(df_day,'Initial')
df_target = uts.compute_target_consumption_features(df_day,'Target')

df = pd.merge(df_filter, df_target, left_on='Id',right_on= 'Participation', how='left')

df = df[df['GoalOfProgram']==1]

df = df[df['Participation'].isna()]


#%% Getting the succesfull ones

    
    X1 = df_temp[df_temp['Phase']==2]
    X2 = df_temp[df_temp['Phase']==6]
    
    frames = [X1,X2]
    X = pd.concat(frames)
    y = (X['Phase']>2).astype('int32').reset_index(drop=True)
    y = ((X['Phase']>2) & (X['Total_achieved']==7))
    X = X[feats_use] 
    
   
    

#df = df_temp[df_temp['Total_achieved']==7]

#How many actually have complete data
for i in range(7,-1,-1):
    print("%d (%0.2f)"%(sum(df_temp['Total_achieved']==i),(sum(df_temp['Total_achieved']==i)*100)/df_temp.shape[0]))
    

print("\n")    
#How many were succesfull
for i in range(7,-1,-1):    
    df_temp['Success'] = df_temp[list_goal].sum(axis=1)==i
    df_succ = df_temp[df_temp['Success']==True]    
    print("%d (%0.2f)"%(sum(df_temp['Success']),sum((df_temp['Success'])*100)/df_temp.shape[0]))


print("\n") 
#Now the distribution of phases of these participants
df_temp['Success'] = df_temp[list_goal].sum(axis=1)==7

df = df_temp[df_temp['Success']]

for i in range(6,0,-1):
    print("%d (%0.2f)"%(sum(df['Phase']==i),(sum(df['Phase']==i)*100)/df.shape[0]))


#%%#Feature engineering for df_partevents, but it does not work

df_partevents['Participation'] = df_partevents['Participation'].astype('int64')

df = pd.merge(df_partevents, df_filter, left_on= 'Participation', right_on ='Id', how='left')

df = df[~df.Id_y.isna()]

df_temp = uts.filter_time_tables(500000, df_filter, df_partevents,'assignment','DateOfEvent')



#%% This is for filtering phase first and so forth, smarter and more maintanable

import utils_data_visualization as uts

df_phasestart = df_phasestart.drop('Id',axis=1)
df_phasestart = pd.merge(df_phasestart,df_program[['Id','Participator']],left_on='Participation',right_on='Id',how='inner')

#future code, filter everything before using
df = uts.filter_phase(df_merge_challenge,'assign', df_phasestart,'Participation','Participation','DateCompletedUtc','DateStarted',2)

df = uts.filter_phase(df_events,'events', df_phasestart,'EventGenerator','Participator','DateOfEvent','DateStarted',2)

df = uts.filter_phase(df_forumpost,'forum', df_phasestart,'Author','Participator','DateCreated','DateStarted',2)


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


#%% Based on the events I check their comsumption

#ids from eventgenerator connects to id from participator (sql confirmed)
df_test = df_participator[df_participator.Id==ids]

#participator column from ProgramParticipation connects to Id from Participator Table, and therefore to eventgenerator
df_s = df_program[df_program.Participator==ids]
#the Id from ProgramParticipation connections to the Participation column in ConsumptionRegistration
part_id = df_s.Id.iloc[0]
#part_id = ids

df_reg_cons = df_cons[df_cons.Participation==part_id]
df_crav_s = df_crav[df_crav.ProgramParticipation==part_id]

df_plot_reg = pd.DataFrame(columns=['Date','Units'])

if df_crav_s.shape[0]!=0:
    df_crav_s['DateOfCraving'] = pd.to_datetime(df_crav_s['DateOfCraving'])
    df_plot_reg['Date'] = df_crav_s['DateOfCraving'].dt.date
    df_plot_reg['Units'] = df_crav_s['NumberOfUnitsConsumed']
    
else:
    df_reg_cons['DateOfRegistration'] = pd.to_datetime(df_reg_s['DateOfRegistration'])
    df_plot_reg['Date'] = df_reg_s['DateOfRegistration'].dt.date
    df_plot_reg['Units'] = df_reg_s['NumberOfUnitsConsumed']

df_plot = pd.merge(df_plot,df_plot_reg,on='Date',how='left')

#%% Creating the plot

df_plot = df_plot.sort_values(by=['Date'])
df_plot = df_plot.fillna(0)

#df_plot = df_plot.drop([79])

sns.set(rc={'figure.figsize':(23,12)})
plot = sns.lineplot(x='Date', y='value', hue='variable', data=pd.melt(df_plot, ['Date']))
#plot = sns.barplot(x='Date', y='value', hue='variable', data=pd.melt(df_plot, ['Date']))
plot.set(xticks=df_plot['Date'].values)
plt.setp(plot.get_xticklabels(), rotation=45)

df_plot.set_index('Date').plot.bar(fontsize='12')


#%% vor checking and visualizing the data since somes frames are huge
#df_merge[df_merge.Participation==14744]

df = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\feature_datasets\Alcohol72.csv")
df2 = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\feature_datasets\Alcoholinf.csv")

ids = df_merge_reg.Participation.iloc[0]

df_vis = df_merge_reg[df_merge_reg['Participation']==ids]

df_merge = df_merge[df_merge['Finished']==1]
df_merge.Id.nunique()

df_vis = df_merge_events.head(10)


df_merge.Id.nunique()



#checking the overlap
set_events = set(df_partevents.Participation)
set_prog_id = set(df_filter.Id)


int1 = set_events.intersection(set_prog_id)
int2 = set_events.intersection(set_prog_part)
act = list(set(df_merge_events['Title']))
act = act[2]



data = {'X':['A', 'A', 'A', 'B', 'B', 'B'],
        'Date' :['2019-06-01', '2019-05-01', '2019-04-01', '2019-06-01', '2019-05-01', '2019-04-01'],
        'Score': [80, 70, 60, 80, 50, 70]}

dfs = pd.DataFrame.from_dict(data)


ax = sns.lineplot(x="Date", y="Score", hue="X", data=df)
plt.show()

set_crav = set(df_crav.ProgramParticipation)
set_reg = set(df_reg.Participation)

int1 = set_crav.intersection(set_reg)

act = list(set(df_merge_events['Title']))
act = act[2]






file1 = open(os.path.join(PATH_DATA,'ForumPost.txt'), 'r') 
lines = file1.readlines()     
file1.close()

new_lines = list()
for i,s in enumerate(lines):
    new_lines.append(s.replace('\x00',''))
    
n_cols = new_lines[0].count('","')   

cols = new_lines[0]
cols = cols.replace('ÿþ','')
#cols = cols.replace('ÿþ','\'')
cols = cols.replace('\n','')
#cols = cols.replace(',','\',\'')
#cols = str(cols + '\'')

cols = cols.split('","')
cols[0] = cols[0].replace('"','')
cols[n_cols] = cols[n_cols].replace('"','')

k = 2

line = new_lines[k].replace('\n','')
line = new_lines[k].split('","')
line[0] = line[0].replace('"','')
line[n_cols] = line[n_cols].replace('"\n','')
df = pd.DataFrame([line], columns = cols)

i = 4

error = list()
lines_error = list()
while i<(len(new_lines)-1):
#while i<200:
    if (new_lines[i].count('","'))==len(cols)-1:
        line = new_lines[i].replace('\n','')
        line = new_lines[i].split('","')
        line[0] = line[0].replace('"','')
        line[n_cols] = line[n_cols].replace('"\n','')
        df2 = pd.DataFrame([line], columns = cols)        
        df = pd.concat([df, df2])        
    else:
        lines_error.append(i)
        error.append(new_lines[i])
    i = i + 2

df.to_csv(os.path.join(os.path.join(PATH_DATA,'ForumPost.csv')),index=False)









k=1
df = pd.DataFrame(columns = cols)
while k<len(new_lines):
    print(k)
    final_line=""
    line=list()
    while len(final_line)<n_cols:
        temp_line = new_lines[k].split('","')
        new_temp_line = list()
        for v in temp_line:
            #new_temp_line.append(v.replace('\n','"?"'))        
            line.append(v.replace('\n','->'))
        k = k + 1
        s = ""
        s = '","'.join([str(elem) for elem in line]) 
        s = s.replace('->","','')
        s = s.replace('->','')
        final_line = s.split('","')
        
    df2 = pd.DataFrame([final_line], columns = cols)
    df = pd.concat([df, df2])  

for v in (temp_line):
    v = v.replace('\n','')
    
    
    
    
#data = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python\Alcohol72.csv")


from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = True,fmt='.1g')


