# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:42:24 2021

@author: laramos
"""
import os
import pandas as pd
from datetime import timedelta


def load_data(path_data,file_name):
    
    _,ext = os.path.splitext(file_name)
    if (ext=='.xls') or (ext=='xlsx'): 
        df = pd.read_excel(os.path.join(path_data,file_name))
    else:
        df = pd.read_csv(os.path.join(path_data,file_name),sep =',' ,encoding = ' ISO-8859-1')
            
    return(df)

def fix_frames(df_program, df_participator,df_phasestart, df_events, df_forumpost, df_forumthread, df_thread):
    
    df_program['StartDateOfParticipation'] = pd.to_datetime(df_program['StartDateOfParticipation'])
    df_program['Finished'] = df_program['Finished']*-1
    df_program['IsActive'] = df_program['IsActive']*-1
    df_program['Activated'] = df_program['Activated']*-1
    df_program['Participator'] = df_program['Participator'].fillna(-1).astype('int64')
    
    df_participator['IsAvailableForResearch'] = df_participator['IsAvailableForResearch']*-1
    
    df_phasestart['DateStarted'] = pd.to_datetime(df_phasestart['DateStarted'])
    
    df_events['EventGenerator'] = pd.to_numeric(df_events['EventGenerator'], errors='coerce').astype(pd.Int64Dtype())
    df_events['DateOfEvent'] = pd.to_datetime(df_events['DateOfEvent'])
    
    df_forumpost['Author'] = df_forumpost['Author'].astype('int64')
    df_forumpost['DateCreated'] = pd.to_datetime(df_forumpost['DateCreated'])
    
    df_forumthread['Author'] = df_forumthread['Author'].astype('int64')
    df_forumthread['DateCreated'] = pd.to_datetime(df_forumthread['DateCreated'])
    
    df_thread['Participator'] = df_thread['Participator'].astype('int64')
    df_thread['Date'] = pd.to_datetime(df_thread['Date'])    
                   
    return(df_program, df_participator, df_phasestart, df_events, df_forumpost,df_forumthread,df_thread)

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
    #This merge is different because there are multiple forum posts and some participants have multiple entries in the 
    #Program table, so we match per date as well
    
    df_activity = df_activity.sort_values(by=[leftdate_on])
    df_program = df_program.sort_values(by=[rightdate_on])
    
    df_program[right_by] = df_program[right_by].fillna(-1)
    df_program[right_by] = df_program[right_by].astype('int64')
    
    df_activity[left_by] = df_activity[left_by].fillna(-1)
    df_activity[left_by] = df_activity[left_by].astype('int64')
    
    df_merge_ = pd.merge_asof(df_activity, df_program, left_on=leftdate_on, right_on = rightdate_on, left_by=left_by, 
                                       right_by = right_by)
    return(df_merge_)

def merge_multactivity_forum(df_activity,df_program, leftdate_on, rightdate_on,left_by,right_by):
    #This merge is different because there are multiple forum posts and some participants have multiple entries in the 
    #Program table, so we match per date as well
    
    df_activity = df_activity.sort_values(by=[leftdate_on])
    df_program = df_program.sort_values(by=[rightdate_on])
        
    df_merge_ = pd.merge_asof(df_activity, df_program, left_on=leftdate_on, right_on = rightdate_on, left_by=left_by, 
                                       right_by = right_by)
    return(df_merge_)

def clean_feats(type_val, pros_cons_dic, df_feats):
    #Splits the features and renames them
    feats = df_feats[df_feats['AnswerType']==type_val]
    feats = feats.drop(['AnswerType'],axis=1)
    feats = feats.rename(columns={"counts": pros_cons_dic[type_val]})
    return(feats)  
 
def match_ids(df_filter,col_id_filter, df_frame,col_id_frame):
    #match ids from 2 frames and return the intersection between them
    ids_filter = (set(df_filter[col_id_filter]))
    ids_using = (set(df_frame[col_id_frame]))
    ids_inter = list(ids_filter.intersection(ids_using))
    return (ids_inter)
    
def bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_merge_feats,feat_name):

    df_startvideo = df_merge_challenge[df_merge_challenge.Name==feat_name]
    
    #ids_filter = (set(df_filter.Id))
    #ids_startvideo = (set(df_startvideo.Participation))
    #ids_inter = list(ids_filter.intersection(ids_startvideo))
    
    df_startvideo = df_startvideo[['Participation','Name','DateCompletedUtc']]
    df_startvideo['Participation'] = df_startvideo['Participation'].fillna(-1)
    df_startvideo['Participation'] = df_startvideo['Participation'].astype('int64')
    df_startvideo['DateCompletedUtc'] = pd.to_datetime(df_startvideo['DateCompletedUtc'])
    
    df_merge_startvideo = merge_multactivity_program(df_startvideo,df_program,'DateCompletedUtc','StartDateOfParticipation','Participation','Id')
    #df_merge_startvideo = df_merge_startvideo.drop_duplicates(subset=['Id','StartDateOfParticipation'])
    
    df_merge_startvideo[feat_name] = (df_merge_startvideo.Name==feat_name).astype('int32')
    df_merge_startvideo = df_merge_startvideo[['Id',feat_name]]
    
    df_temp  = df_merge_startvideo.groupby(['Id']).agg({feat_name:['sum']})#.unstack()
    df_merge_startvideo = pd.DataFrame(columns = ['Id',feat_name])
    df_merge_startvideo['Id'] = df_temp.index
    df_merge_startvideo[feat_name] = df_temp.values
    
    
    df_temp = pd.merge(df_merge_feats,df_merge_startvideo,left_on = 'Id', right_on='Id',how = 'left')
    
    return(df_temp)

def convert_dttime_to_hours(df,diff_col,new_col):
    days = (df[diff_col].dt.days)*24
    hours = (df[diff_col].dt.seconds)/3600
    total_time = days + hours
    df[new_col] = total_time
    return(df)
                            

def filter_time_tables(time_hours,df_filter,df_frame, type_frame, name_date):
    #function usef to filter assignments done within a certain time after start
    df_temp = df_filter[['Id','StartDateOfParticipation','GoalOfProgram', 'Participator']]
    df_temp['Id'] = df_temp['Id'].astype('int64')
    
    if type_frame=='assignment':       
        df = pd.merge(df_frame, df_temp, left_on= 'Participation', right_on ='Id', how='left')
    elif type_frame=='forum':          
        df = merge_multactivity_program(df_frame,df_temp,name_date,'StartDateOfParticipation','Author','Participator')       
        df = df[df.Participator.notna()]   
    elif type_frame=='events': 
        df_frame['EventGenerator'] = df_frame['EventGenerator'].astype('int64')           
        df = merge_multactivity_program(df_frame,df_temp,name_date,'StartDateOfParticipation','EventGenerator','Participator')
    elif type_frame=='thread':        
        df = merge_multactivity_program(df_frame,df_temp,name_date,'StartDateOfParticipation','Participator','Participator')
            
            #df = pd.merge(df_frame, df_temp, left_on= 'EventGenerator', right_on ='Participator', how='left')
    df[name_date] = pd.to_datetime(df[name_date])
    df['Diff'] = df[name_date] - df['StartDateOfParticipation']
    df = convert_dttime_to_hours(df,'Diff','Total time')

    df_temp = df[df['Total time']<=time_hours]
    return df_temp


def compute_target_consumption_features(df_day,target_or_consumption):
    
    df_temp = df_day[df_day['Type']==target_or_consumption]
    
    df_temp  = df_temp.groupby(['Participation','DayOfWeek']).agg({'Value':['sum']}).unstack()
    
    df_feats = pd.DataFrame()

    df_feats['Participation'] = df_temp['Value']['sum'][0].index
    df_feats['Monday' + target_or_consumption] = df_temp['Value']['sum'][0].values
    df_feats['Tuesday' + target_or_consumption] = df_temp['Value']['sum'][1].values
    df_feats['Wednesday' + target_or_consumption] = df_temp['Value']['sum'][2].values
    df_feats['Thursday' +  target_or_consumption] = df_temp['Value']['sum'][3].values
    df_feats['Friday' + target_or_consumption] = df_temp['Value']['sum'][4].values
    df_feats['Saturday' + target_or_consumption] = df_temp['Value']['sum'][5].values
    df_feats['Sunday' + target_or_consumption] = df_temp['Value']['sum'][6].values
    
    return (df_feats)

def filter_phase(df_frame,type_frame,df_phase,id_frame,id_phase,time_frame,time_phase,num_phase):
    
    df_phase_filter = df_phase[df_phase['Phase']==num_phase]
    
    df_phase_filter = df_phase_filter.drop(['Id','Phase'],axis=1)
    
    if type_frame=='assign':
        df = pd.merge(df_frame,df_phase_filter, left_on=id_frame,right_on=id_phase,how='left')
    else:
        #df_frame[id_frame] = df_frame[id_frame].astype('int64')   
        #df_phase[id_phase] = df_phase[id_phase].astype('int64')          
        df = merge_multactivity_program(df_frame,df_phase_filter, time_frame,time_phase,id_frame,id_phase)   
        df[time_phase] = df [time_phase].fillna(df [time_frame])
    
    df[time_frame] = pd.to_datetime(df[time_frame])
    
    df['After Phase']= df[time_frame]>df[time_phase]
    
    df_frame = df
    
    df_frame = df[df['After Phase']==False]
    df_frame = df_frame.drop([time_phase,'After Phase'],axis=1)
    
    return (df_frame)
    
    
    

def select_participants_from_phase(df_program, PROGRAM, df_merge_challenge,df_phasestart):
    #Paul suggested we only use data from challenge since it is the new program. Participants from old program dont have phase either
    df_filter = df_program[df_program.Type=='Challenge']
    df_filter = df_filter[df_filter['Program']==PROGRAM]
    #df_filter = df_filter[df_filter['Phase'].isin(phases)]
    
    #Getting only phase 1 because we just need the features from phase 1 for now
    df_merge_challenge = df_merge_challenge[df_merge_challenge['Phase']==1]
    
    #here I check the assingments from phase 1 that were finished after the start of phase 2
    #get phase 2 related information
    df_phase_filter = df_phasestart[df_phasestart.Phase==2]
    df_phase_filter = df_phase_filter.drop(['Id','Phase'],axis=1)
    
    df = pd.merge(df_merge_challenge,df_phase_filter, on='Participation',how='left')
    df['DateCompletedUtc'] = pd.to_datetime(df['DateCompletedUtc'])
    
    df['After Phase 2']= df.DateCompletedUtc>df.DateStarted
    
    df_merge_challenge = df[df['After Phase 2']==False]
    
   # print("Assignment from Program: %s selected!"%(program_dict[PROGRAM]))
    
    return (df_merge_challenge,df_filter)


def filter_assigment_ids_after_phase(assign_name,df_merge_challenge,df_phasestart,df_filter):
    
    df_merge_challenge = df_merge_challenge[df_merge_challenge['Phase']==1]

    df = df_merge_challenge[df_merge_challenge.Name.isin(assign_name)]
    df2 = df.groupby(['Participation']).size().reset_index(name='Number')
    df2 = df2[df2['Number']>1]
    
    ids = df2.Participation
    df = df[df.Participation.isin(ids)]
    df_phase2 = df_phasestart[df_phasestart.Phase==2][['Phase', 'DateStarted', 'Participation']]
    
    df = pd.merge(df,df_phase2, on='Participation', how ='left')
    df = df[df.Phase_y.notna()]
    df['DateCompletedUtc'] = pd.to_datetime(df['DateCompletedUtc'])
    df['diff'] = df['DateStarted'] - df['DateCompletedUtc']
    
    df2 = df[df['diff'].dt.days<0]
    ids_after_phase1 = list(set(df2.Participation))
    
    df_merge_challenge = df_merge_challenge[~df_merge_challenge.Participation.isin(ids_after_phase1)]
    df_filter = df_filter[~df_filter.Id.isin(ids_after_phase1)]
    return(df_merge_challenge,df_filter)

def feature_engineering_pros_cons(df_filter,df_pcusing,pros_cons_dic):
             
    ids_match = match_ids(df_filter,'Id', df_pcusing,'Participation')

    df_temp = df_pcusing[df_pcusing.Participation.isin(ids_match)]
    df_temp['Answer Length'] = df_temp['Answer'].str.len()    
    
    df_feats = df_temp.groupby(['Participation', 'AnswerType']).size().reset_index(name='counts')    
    
    pros_short = clean_feats(3,pros_cons_dic,df_feats)
    pros_long = clean_feats(4,pros_cons_dic,df_feats)
    cons_short = clean_feats(5,pros_cons_dic,df_feats)
    cons_long = clean_feats(6,pros_cons_dic,df_feats)
    
    #df_feats['Total Answer Length'] = df_temp.groupby(['Participation','AnswerType']).agg({'Answer Length':['sum']})['Answer Length']['sum'].values
    
    merge_feats = pd.merge(pros_short,pros_long,on='Participation',how = 'outer')
    merge_feats = pd.merge(merge_feats,cons_short,on='Participation', how= 'outer')
    merge_feats = pd.merge(merge_feats,cons_long,on='Participation',  how = 'outer')
    #merge_feats = pd.merge(merge_feats,df_feats[['Participation','Total Answer Length']],on='Participation',  how = 'left')
    
    df_filter_merge = df_filter[['Id','StartDateOfParticipation','Participator','Program','GoalOfProgram','Phase']]
    #here we select which columns will be included in the feature variable
    df_merge_feats = pd.merge(df_filter_merge[['Id','Phase','StartDateOfParticipation','GoalOfProgram']],merge_feats,left_on = 'Id', right_on='Participation',how = 'left')
    
    return (df_merge_feats)

def feature_engineering_binary(df_merge_challenge,df_filter,df_program,df_merge_feats):
    
    df_temp = bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_merge_feats,'Start video')
    df_temp = bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_temp,'Afspraken maken')
    df_temp = bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_temp,'Voor- en nadelen')
    df_merge_feats = bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_temp,'Jouw afspraken')
    
    return (df_merge_feats)

def feature_engineering_assignment_agreement(df_filter,df_merge_feats,df_assign_answer,pros_cons_dic):
    
    ids_match = match_ids(df_filter,'Id', df_assign_answer,'Participation')
    
    df_temp = df_assign_answer[df_assign_answer.Participation.isin(ids_match)]
    df_temp['Agreement Length'] = df_assign_answer['Agreement'].str.len()
    
    df_feats = df_temp.groupby(['Participation']).size().reset_index(name='Number of Agreements')
    df_feats['Total Agreement Length'] = df_temp.groupby(['Participation']).agg({'Agreement Length':['sum']})['Agreement Length']['sum'].values
    
    df_merge_feats = df_merge_feats.drop(['Participation'],axis=1)
    
    #here we select which columns will be included in the feature variable
    df_merge_feats = pd.merge(df_merge_feats,df_feats,left_on = 'Id', right_on='Participation',how = 'left')
    
    #in case hours was used to filter, some participants will have pros and cons but the voo-en nadelen will be set to nan, so we remove pros and cons 
    
    for key in pros_cons_dic.keys():    
        df_merge_feats[pros_cons_dic[key]] = df_merge_feats['Voor- en nadelen']*df_merge_feats[pros_cons_dic[key]]
    
    return(df_merge_feats)

def save_frame(frame,path):
    frame.to_csv(path,index=False)


def feature_engineering_consumption_features(df_day,df_merge_feats):
            
    df_initial = compute_target_consumption_features(df_day,'Initial')
    df_target = compute_target_consumption_features(df_day,'Target')
     
    df_feats_cons = pd.merge(df_initial, df_target,on = 'Participation')
    print(df_feats_cons.columns)
    
    df_merge_feats = pd.merge(df_merge_feats,df_feats_cons, left_on= 'Id',right_on = 'Participation',how = 'left')
    df_merge_feats = df_merge_feats.drop(['Participation_x','Participation_y'],axis=1)
    df_feats_cons = df_feats_cons.drop(['Participation'],axis=1)
    
    feats = df_feats_cons.columns
    df_merge_feats[feats] = df_merge_feats[feats].fillna(-1)
    # df_temp['tot'] = np.sum(df_temp[feats[1:len(feats)]])
    # df_temp = df_temp[df_temp]
    return(df_merge_feats)


def create_features_time(df,feat_name,time_name,new_feat_name):
    df_feat = df[df['Name']==feat_name][['Participation',time_name]]
    df_feat = convert_dttime_to_hours(df_feat,time_name,new_feat_name)
    df_feat = df_feat.rename(columns={'Participation':'Id'})
    df_feat = df_feat.drop(time_name,axis=1)
    return (df_feat)

def feature_engineering_time_assigment(df_merge_challenge_feat,df_merge_feats,df_program,date_start,date_finished):

    df_merge_challenge_feat = pd.merge(df_merge_challenge_feat,df_program[['Id',date_start]],left_on = 'Participation', right_on='Id')
    df_merge_challenge_feat['Complete Time'] = df_merge_challenge_feat[date_finished] - df_merge_challenge_feat[date_start]
    #df_merge_challenge_feat = df_merge_challenge_feat.drop(['Id_x','Id_y'],axis=1)
    
    df = df_merge_challenge_feat.groupby(['Participation','Name']).first()
    #df['Name'] = df.index.get_level_values(1)
    df = df.reset_index(level=['Participation'])
    df = df.reset_index(level=['Name'])
    
    df_video = create_features_time(df,'Start video','Complete Time','StartVideo_time')
    df_voorennadelen = create_features_time(df,'Afspraken maken','Complete Time','AfsprakenMaken_time')
    df_afspraak = create_features_time(df,'Voor- en nadelen','Complete Time','ProsCons_time')
    df_jouw= create_features_time(df,'Jouw afspraken','Complete Time','Jouw_afspraken_time')    
    
    df_merge_feats = pd.merge(df_merge_feats,df_video,on='Id',how='left')
    df_merge_feats = pd.merge(df_merge_feats,df_voorennadelen,on = 'Id',how='left')
    df_merge_feats = pd.merge(df_merge_feats,df_afspraak,on = 'Id',how='left')
    df_merge_feats = pd.merge(df_merge_feats,df_jouw,on = 'Id',how='left')
    return(df_merge_feats)


def feature_engineering_forum_and_login_events(TIME_HOURS,df_filter,df_merge_feats,df_events):
    
    df_temp = filter_time_tables(TIME_HOURS, df_filter, df_events,'events','DateOfEvent')
    
    df_temp_visit = df_temp[df_temp['Title']=='Forum Visited']
    df_temp_visit = df_temp_visit.groupby(['Id_y']).size().reset_index(name='Number of Forum Visits')
    
    df_merge_feats = pd.merge(df_merge_feats,df_temp_visit,left_on = 'Id', right_on='Id_y',how = 'left')
    df_merge_feats = df_merge_feats.drop(['Id_y'],axis=1)
    
    df_temp_login = df_temp[df_temp['Title']=='User Login']
    df_temp_login = df_temp_login.groupby(['Id_y']).size().reset_index(name='Number of Logins')
    df_merge_feats = pd.merge(df_merge_feats,df_temp_login,left_on = 'Id', right_on='Id_y',how = 'left')
    df_merge_feats = df_merge_feats.drop(['Id_y'],axis=1)
    
    return(df_merge_feats)

def feature_engineering_forumpost(TIME_HOURS,df_filter,df_forumpost,df_merge_feats,time_forum, var_name):

    df_temp = filter_time_tables(TIME_HOURS, df_filter, df_forumpost,'forum','DateCreated')
    #df_temp = uts.filter_time_tables(72, df_filter, df_forumpost,'forum','DateCreated')
    
    df2 = df_temp.groupby(['Id_y']).size().reset_index(name=var_name)
    df2.Id_y = df2.Id_y.astype('int32')
    
    df_merge_feats = pd.merge(df_merge_feats, df2, left_on = 'Id',right_on = 'Id_y',how = 'left')
    df_merge_feats = df_merge_feats.drop(['Id_y'],axis=1)

    return (df_merge_feats)


def remove_ids_future_edits(df_merge_challenge,df_filter,df_merge_feats,time_hours):
    #This function can be used to remove ids from people that go back to assignments and might edit or add things after the time frame we defined
    #Leaving these participants can include a severe bias
    
    #include other variables that might be affected by this here
    names = ['Voor- en nadelen','Jouw afspraken']
     
    df_merge_challenge_feat = df_merge_challenge[df_merge_challenge['Phase']==1]
    
    df_temp = df_filter[['Id','StartDateOfParticipation','GoalOfProgram', 'Participator']]
    df_temp['Id'] = df_temp['Id'].astype('int64')
    
    df = pd.merge(df_merge_challenge_feat, df_temp, left_on= 'Participation', right_on ='Id', how='left')
              
    df['DateCompletedUtc'] = pd.to_datetime(df['DateCompletedUtc'])
    df['Diff'] = df['DateCompletedUtc'] - df['StartDateOfParticipation']
    df = convert_dttime_to_hours(df,'Diff','Total time')
    
    df_temp = df[df['Total time']<=time_hours]
    df_temp = df[df['Total time']>=time_hours]
    

    df = df_temp[df_temp.Name.isin(names)]
    
    ids1 = (set(df.Participation))
    ids2 = (set(df_merge_feats.Id))
    
    inter = list(ids1.intersection(ids2))
    
    df = df_merge_feats[~df_merge_feats.Id.isin(inter)]
    
    return(df)

def add_consumption_before_last_phase(number_of_days_before, df_cons, df_phasestart,df_merge_feats):

    #We use this variable to get the last 7 days
    days_subtract = timedelta(days=number_of_days_before)

    df_cons6 = df_cons

    #df_phase6 = df_phasestart[df_phasestart.Phase==6][['Phase', 'DateStarted', 'Participation']]
    #get the last phase they achieved
    df_phasegroup = df_phasestart.sort_values('DateStarted').groupby('Participation').tail(1)[['Phase', 'DateStarted', 'Participation']]

    df_phasegroup['date_7days'] = df_phasegroup['DateStarted'] - days_subtract

    df_merge_phasecons = pd.merge(df_cons6,df_phasegroup,on='Participation',how='left')
    
    #These participants did not reach further than phase 1
    df_merge_phasecons = df_merge_phasecons[df_merge_phasecons['date_7days'].notna()]

    df_merge_phasecons['date_7days'] = pd.to_datetime(df_merge_phasecons['date_7days'])
    df_merge_phasecons['DateOfRegistration'] = pd.to_datetime(df_merge_phasecons['DateOfRegistration'])

    # Here I get the interval between their last phase and 7 days before
    df_merge_phasecons = df_merge_phasecons[df_merge_phasecons['DateOfRegistration']>df_merge_phasecons['date_7days']]
    df_merge_phasecons = df_merge_phasecons[df_merge_phasecons['DateOfRegistration']<df_merge_phasecons['DateStarted']]

    df = df_merge_phasecons.copy()

    df['DayOfWeek'] = df['DateOfRegistration'].dt.day_name()
    #Some participants register 
    df_temp = df.groupby(['Participation','DayOfWeek']).agg({'NumberOfUnitsConsumed':['sum']}).unstack()
    df_temp = df_temp.reset_index(level=['Participation'])
    df_temp = df_temp.droplevel(1,axis=1)
    df_temp = df_temp.droplevel(0,axis=1)
    cols = list(df_temp.columns)
    cols[0] = 'Participation'
    df_temp = pd.DataFrame(df_temp.values,columns = cols)

    #df_temp = df_temp.dropna(how='any',axis=0)
    df_temp  = pd.merge(df_merge_feats,df_temp,left_on ='Id',right_on = 'Participation',how='left')
    df_temp = df_temp.drop('Participation',axis=1)
    
    return(df_temp)



def feature_engineering_thread_vieweing(TIME_HOURS,df_thread,df_filter,df_merge_feats):

    df_temp = df_filter.copy()
    
    df_temp = filter_time_tables(TIME_HOURS, df_temp, df_thread,'thread','Date')
    
    df_temp = df_temp.groupby(['Id_y']).size().reset_index(name='Number of Thread Views')
    
    df_temp  = pd.merge(df_merge_feats,df_temp,left_on ='Id',right_on = 'Id_y',how='left')
    
    df_temp = df_temp.drop('Id_y',axis=1)
    
    return(df_temp)


def feature_engineering_badge(TIME_HOURS,df_filter,df_partbadge,df_merge_feats):
    
    df_temp = filter_time_tables(TIME_HOURS, df_filter, df_partbadge,'assignment','DateAssignedUtc')
    
    df_temp = df_temp.groupby(['Participation']).size().reset_index(name='Number of Participation Badges')
    
    df_temp  = pd.merge(df_merge_feats,df_temp,left_on ='Id',right_on = 'Participation',how='left')
    
    df_temp = df_temp.drop('Participation',axis=1)

    return(df_temp)


def feature_engineering_achievementlike(TIME_HOURS,df_filter,df_achievelike,df_merge_feats):
    
        df_achievelike = df_achievelike.dropna(subset=['DateReadUtc'])
        
        df_achievelike['DateReadUtc'] = pd.to_datetime(df_achievelike['DateReadUtc'])    
        
        df_temp = filter_time_tables(TIME_HOURS,df_filter,df_achievelike, 'thread', 'DateReadUtc')
        
        df_temp = df_temp.rename(columns={'Id_y':'Id'})
        
        df_temp = df_temp.groupby(['Id']).size().reset_index(name='Number of Achievement Likes')
        
        df = pd.merge(df_merge_feats,df_temp,on='Id',how = 'left')
        
        return(df)
    
    
    
def feature_engineering_consumption_first_days(time_hours, df_cons, df_filter,df_merge_feats):

    #We use this variable to get the last 7 days
    df_merge_cons = pd.merge(df_cons,df_filter,left_on='Participation',right_on='Id',how = 'left')
    
    df_merge_cons = df_merge_cons[~df_merge_cons.Id_y.isna()]

    df_merge_cons['DateSaved'] = pd.to_datetime(df_merge_cons['DateSaved'])
    
    df_merge_cons['Diff'] = df_merge_cons['DateSaved'] - df_merge_cons['StartDateOfParticipation']
    
    df = df_merge_cons[['Participation','DateOfRegistration','DateSaved','NumberOfUnitsConsumed','StartDateOfParticipation','Diff']]
    
    df = convert_dttime_to_hours(df,'Diff','Total time')
    
    df = df[df['Total time']<time_hours]
    
    df_temp = df.groupby(['Participation']).agg({'NumberOfUnitsConsumed':['sum']})#.unstack()

    df_frame = pd.DataFrame(columns = ['Id','Total_interval_consumption'])
    df_frame['Id'] = df_temp.index
    df_frame['Total_interval_consumption'] = df_temp.values
    
    df_temp  = pd.merge(df_merge_feats,df_frame,on = 'Id',how='left')
    return (df_temp)

