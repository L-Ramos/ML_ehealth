# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:25:16 2020

@author: laramos

To Do:
SystemEvents  
eventgenerators are missing, we can infer from ip if avaiable (can we? must ask)

Function to filter substances and goals

"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

PATH_DATA = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python"


def load_data(path_data,file_name):
    
    _,ext = os.path.splitext(file_name)
    if (ext=='.xls') or (ext=='xlsx'): 
        df = pd.read_excel(os.path.join(PATH_DATA,file_name))
    else:
        df = pd.read_csv(os.path.join(PATH_DATA,file_name),sep =',' ,encoding = ' ISO-8859-1')
            
    return(df)

def fix_frames(df_program, df_participator,df_phasestart, df_events, df_forumpost):
    
    df_program['StartDateOfParticipation'] = pd.to_datetime(df_program['StartDateOfParticipation'])
    df_program['Finished'] = df_program['Finished']*-1
    df_program['IsActive'] = df_program['IsActive']*-1
    df_program['Activated'] = df_program['Activated']*-1
    
    df_participator['IsAvailableForResearch'] = df_participator['IsAvailableForResearch']*-1
    
    df_phasestart['DateStarted'] = pd.to_datetime(df_phasestart['DateStarted'])
    
    df_events['EventGenerator'] = pd.to_numeric(df_events['EventGenerator'], errors='coerce').astype(pd.Int64Dtype())
    df_events['DateOfEvent'] = pd.to_datetime(df_events['DateOfEvent'])
    
    df_forumpost['Author'] = df_forumpost['Author'].astype('int64')
    df_forumpost['DateCreated'] = pd.to_datetime(df_forumpost['DateCreated'])
               
    return(df_program, df_participator, df_phasestart, df_events, df_forumpost)

def filter_date(df,date_column,date_to_filter='2016-01-01'):
    df_filter = df[(df[date_column] > date_to_filter)]
    #print("Unique Ids in ProgramParticipation that started after 2016 (start date of ConsumptionRegistration): ",df_filter.Id.nunique())
    return(df_filter)
    
def filter_program_and_goal(df_program, n_progr):

    if n_progr==0:
        df_prog = df_program
    else:
        df_prog = df_program[df_program['Program']==n_progr]

    df_prog_stop = df_prog[df_prog['GoalOfProgram']==1]
    df_prog_red = df_prog[df_prog['GoalOfProgram']==2]
    df_prog_s_stop = df_prog[df_prog['GoalOfProgram']==3]
    df_prop_s_red = df_prog[df_prog['GoalOfProgram']==4]
    
    return(df_prog,df_prog_stop,df_prog_red,df_prog_s_stop,df_prop_s_red)
      
def merge_multactivity_program(df_activity,df_program, leftdate_on, rightdate_on,left_by,right_by):
    
    df_activity = df_activity.sort_values(by=[leftdate_on])
    df_program = df_program.sort_values(by=[rightdate_on])
    
    df_program['Participator'] = df_program['Participator'].fillna(-1)
    df_program['Participator'] = df_program['Participator'].astype('int64')
    
    df_merge_ = pd.merge_asof(df_activity, df_program, left_on=leftdate_on, right_on = rightdate_on, left_by=left_by, 
                                       right_by = right_by)
    return(df_merge_)
    

#%% Loading Excel files

df_participator = load_data(PATH_DATA,"Participator.xls") 

df_program = load_data(PATH_DATA,"ProgramParticipation.xls") 

df_reg = load_data(PATH_DATA,"ConsumptionRegistration.xls") 

df_crav = load_data(PATH_DATA,"CravingAndConsumptionRecord.xls") 

df_phasestart = load_data(PATH_DATA,"PhaseStart.xls") 

df_events = load_data(PATH_DATA,"SystemEvent2.txt") 

#AssignmentCompletion has information about ChallengeAssignment, they complete each other
df_assign = load_data(PATH_DATA, "AssignmentCompletion.xls")
df_challenge = load_data(PATH_DATA, "ChallengeAssignment.xls")

df_assign_read = load_data(PATH_DATA, "ReadReadingAssignment.xls")
df_assign_risk = load_data(PATH_DATA, "AssignmentRiskSituationAnswer.xls")


df_forumpost = load_data(PATH_DATA,"ForumPost.xls")

df_pcstop = load_data(PATH_DATA, "AssignmentAnswerProsConsStopping.csv")

df_pcusing = load_data(PATH_DATA, "AssignmentAnswerProsConsUsing.csv")

df_assign_answer = load_data(PATH_DATA, "AssignmentAnswerAgreements.csv")

df_program, df_participator, df_phasestart, df_events, df_forumpost = fix_frames(df_program, df_participator, df_phasestart, df_events, df_forumpost)

print("Data Laoded")

#FIX THIS
#eventgenerators are missing, we can infer from ip if avaiable
df_events = df_events.dropna(subset=['EventGenerator'])
#print("Unique Ids in SystemEvent: ",df_events.EventGenerator.nunique())


#%% Filtering by date, so we include only after a certain year

df_participator = filter_date(df_participator,'DateCreated')
df_program = filter_date(df_program,'StartDateOfParticipation')
df_reg = filter_date(df_reg,'DateOfRegistration')
df_crav = filter_date(df_crav,'DateOfCraving')
df_phasestart = filter_date(df_phasestart,'DateStarted')
df_events = filter_date(df_events,'DateOfEvent')
df_forumpost = filter_date(df_forumpost,'DateCreated')

print('Tables filtered by Date.')


#%% Merge both ProgramParticipation with the rest

df_merge = df_program

df_merge_part = pd.merge(df_participator,df_merge,left_on='Id',right_on='Participator')
print(df_participator.shape[0],df_merge_part.shape[0])

df_merge_reg = pd.merge(df_reg,df_merge,left_on='Participation',right_on='Id')
print(df_reg.shape[0],df_merge_reg.shape[0])

#df_merge_crav = pd.merge(df_crav,df_merge,left_on='ProgramParticipation',right_on='Id')

df_merge_events = pd.merge(df_events,df_merge,left_on='EventGenerator',right_on='Id',how='inner')
print(df_events.shape[0],df_merge_events.shape[0])

df_merge_assign = pd.merge(df_assign,df_merge,left_on='Participation',right_on='Id',how='inner')
print(df_merge_assign.shape[0],df_assign.shape[0])

df_merge_challenge = pd.merge(df_assign,df_challenge,left_on='Assignment',right_on='Id',how='inner')
print(df_assign.shape[0],df_merge_challenge.shape[0])

df_merge_forumpost = merge_multactivity_program(df_forumpost,df_program,'DateCreated','StartDateOfParticipation','Author','Id')
print(df_forumpost.shape[0],df_merge_forumpost.shape[0])

#df_merge_risk = pd.merge(df_assign,df_assign_risk,left_on='Assignment',right_on='Id',how='inner')
print("Tables merged.")
#%% Selecting only Participants from Alcohol Phase 1 and > 2
def clean_feats(type_val, pros_cons_dic, df_feats):
    #Splits the features and renames them
    feats = df_feats[df_feats['AnswerType']==type_val]
    feats = feats.drop(['AnswerType'],axis=1)
    feats = feats.rename(columns={"counts": pros_cons_dic[type_val]})
    return(feats)
#df_merge_challenge  = df_merge_challenge.drop(['Id_x'])
#I dont have to filter by phase 1 and 2, becayse people at phase 3 also did phase 2
#phases = [1,2]

#Paul suggested we only use data from challenge since it is the new program. Participants from old program dont have phase either
df_filter = df_program[df_program.Type=='Challenge']
df_filter = df_filter[df_filter['Program']==1]
#df_filter = df_filter[df_filter['Phase'].isin(phases)]

#Getting only phase 1 because we just need the features from phase 1 for now
df_merge_challenge = df_merge_challenge[df_merge_challenge['Phase']==1]

#here I check the assingments from phase 1 that were finished after the start of phase 2
df_phase_filter = df_phasestart[df_phasestart.Phase==2]
df_phase_filter = df_phase_filter.drop(['Id','Phase'],axis=1)

df = pd.merge(df_merge_challenge,df_phase_filter, on='Participation',how='left')
df['DateCompletedUtc'] = pd.to_datetime(df['DateCompletedUtc'])

df['After Phase 2']= df.DateCompletedUtc>df.DateStarted

df_merge_challenge = df[df['After Phase 2']==False]

#%% Feature engineering for pros and cons

#AnswerType
#cons 5 (short term)  6(long term)
#pros 3 (short term) 4(long term)

ids_filter = (set(df_filter.Id))
ids_using = (set(df_pcusing.Participation))
ids_inter = list(ids_filter.intersection(ids_using))


df_temp = df_pcusing[df_pcusing.Participation.isin(ids_inter)]
df_temp['Answer Length'] = df_temp['Answer'].str.len()

pros_cons_dic = {3:'Pros Short', 4: 'Pros Long', 5: 'Cons Short', 6: 'Cons Long'} 

df_feats = df_temp.groupby(['Participation', 'AnswerType']).size().reset_index(name='counts')
#df_feats['Total Answer Length'] = df_temp.groupby(['Participation','AnswerType']).agg({'Answer Length':['sum']})['Answer Length']['sum'].values

pros_short = clean_feats(3,pros_cons_dic,df_feats)
pros_long = clean_feats(4,pros_cons_dic,df_feats)
cons_short = clean_feats(5,pros_cons_dic,df_feats)
cons_long = clean_feats(6,pros_cons_dic,df_feats)


merge_feats = pd.merge(pros_short,pros_long,on='Participation',how = 'outer')
merge_feats = pd.merge(merge_feats,cons_short,on='Participation', how= 'outer')
merge_feats = pd.merge(merge_feats,cons_long,on='Participation',  how = 'outer')
#merge_feats = pd.merge(merge_feats,df_feats[['Participation','Total Answer Length']],on='Participation',  how = 'outer')

df_filter_merge = df_filter[['Id','StartDateOfParticipation','Participator','Program','GoalOfProgram','Phase']]
#here we selct which columns will be included in the feature variable
df_merge_feats = pd.merge(df_filter_merge[['Id','Phase','StartDateOfParticipation','GoalOfProgram']],merge_feats,left_on = 'Id', right_on='Participation',how = 'left')


#%% Feature engineering for binary features from phase 1

def feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_merge_feats,feat_name):

    df_startvideo = df_merge_challenge[df_merge_challenge.Name==feat_name]
    
    ids_filter = (set(df_filter.Id))
    ids_startvideo = (set(df_startvideo.Participation))
    ids_inter = list(ids_filter.intersection(ids_startvideo))
    
    df_startvideo = df_startvideo[['Participation','Name','DateCompletedUtc']]
    df_startvideo['Participation'] = df_startvideo['Participation'].fillna(-1)
    df_startvideo['Participation'] = df_startvideo['Participation'].astype('int64')
    df_startvideo['DateCompletedUtc'] = pd.to_datetime(df_startvideo['DateCompletedUtc'])
    
    df_merge_startvideo = merge_multactivity_program(df_startvideo,df_program,'DateCompletedUtc','StartDateOfParticipation','Participation','Id')
    df_merge_startvideo = df_merge_startvideo.drop_duplicates(subset=['Id','StartDateOfParticipation'])
    
    df_merge_startvideo[feat_name] = (df_merge_startvideo.Name==feat_name).astype('int32')
    df_merge_startvideo = df_merge_startvideo[['Id',feat_name]]
    
    df_temp = pd.merge(df_merge_feats,df_merge_startvideo,left_on = 'Id', right_on='Id',how = 'left')
    
    return(df_temp)

df_temp = feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_merge_feats,'Start video')
df_temp = feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_temp,'Afspraken maken')
df_temp = feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_temp,'Voor- en nadelen')
df_merge_feats = feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_temp,'Jouw afspraken')

#%% Feature Engineering for AssignmentAnswerAgreements (length of reply / number of replies)
ids_filter = (set(df_filter.Id))
ids_using = (set(df_assign_answer.Participation))
ids_inter = list(ids_filter.intersection(ids_using))

df_temp = df_assign_answer[df_assign_answer.Participation.isin(ids_inter)]
df_assign_answer['Agreement Length'] = df_assign_answer['Agreement'].str.len()

df_feats = df_temp.groupby(['Participation']).size().reset_index(name='Number of Agreements')
df_feats['Total Agreement Length'] = df_temp.groupby(['Participation']).agg({'Agreement Length':['sum']})['Agreement Length']['sum'].values


df_merge_feats = df_merge_feats.drop(['Participation'],axis=1)

#here we selct which columns will be included in the feature variable
df_temp= pd.merge(df_merge_feats,df_feats,left_on = 'Id', right_on='Participation',how = 'left')

df_temp.to_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python\features.csv",index=False)


#%% Feature engineering for Forum



#%% Plotting some events from a participant using only the events table

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

df_reg_s = df_reg[df_reg.Participation==part_id]
df_crav_s = df_crav[df_crav.ProgramParticipation==part_id]

df_plot_reg = pd.DataFrame(columns=['Date','Units'])

if df_crav_s.shape[0]!=0:
    df_crav_s['DateOfCraving'] = pd.to_datetime(df_crav_s['DateOfCraving'])
    df_plot_reg['Date'] = df_crav_s['DateOfCraving'].dt.date
    df_plot_reg['Units'] = df_crav_s['NumberOfUnitsConsumed']
    
else:
    df_reg_s['DateOfRegistration'] = pd.to_datetime(df_reg_s['DateOfRegistration'])
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

ids = df_merge_reg.Participation.iloc[0]

df_vis = df_merge_reg[df_merge_reg['Participation']==ids]

df_merge = df_merge[df_merge['Finished']==1]
df_merge.Id.nunique()

df_vis = df_merge_events.head(10)


df_merge.Id.nunique()



#checking the overlap
set_events = set(df_events.EventGenerator)
set_prog_id = set(df_program.Id)
set_prog_part = set(df_program.Participator)

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