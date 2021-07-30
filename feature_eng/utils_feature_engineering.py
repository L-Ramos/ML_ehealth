# -*- coding: utf-8 -*-
"""

This code offers support fucntions for the feature_engineering.py code

Created on Wed Dec  9 12:35:16 2020

@author: laramos

TODO
convert all hardcoded column names to constants and variables

"""
import os
import pandas as pd
from datetime import timedelta







def load_data(path_data,file_name):
    """ Loads data into a pandas dataframe, either from .xls or .xlsx or .csv format
    
    Arguments:
        path_data (string): folder where the file is
        file_name (string): name of the file to be read, must include extension
        
    Returns:
        df (DataFrame): dataframe with all data loaded               
    """
    
    # check what is the extension of the file to use the right loading function
    _,ext = os.path.splitext(file_name)
    if (ext=='.xls') or (ext=='xlsx'): 
        df = pd.read_excel(os.path.join(path_data,file_name))
    else:
        df = pd.read_csv(os.path.join(path_data,file_name),sep =',' ,encoding = ' ISO-8859-1')
            
    return(df)

def save_frame(frame,path):
    """ Save a dataframe as .csv
    
    Arguments:
        frame (DataFrame): frame to be saved
        path (string): path to be saved to, should contain file name and .csv extension
       
    """
    frame.to_csv(path,index=False)


def convert_dttime_to_hours(df,diff_col,new_col):
    """ convert datetime columns (resulting from a diff statement in pandas) to (int) hours, so they can be easily compared
    
    Arguments:
        df (DataFrame): Frame with the column that needs to be changed
        diff_col (string): name of the column to be converted
        new_col (string): new name of the converted column
        
    Returns:
        df (DataFrame): dataframe with the new column converted              
    """
    
    days = (df[diff_col].dt.days)*24
    hours = (df[diff_col].dt.seconds)/3600
    total_time = days + hours
    df[new_col] = total_time
    
    return(df)    

def fix_frames(df_program, df_participator,df_phasestart, df_events, df_forumpost, df_forumthread, df_thread):
    """ First function to be called after loading the data, fixes types and encoding issues in the data
    
    Arguments:
        df_program (DataFrame): Frame with information about the user and each program
        df_participator (DataFrame): Frame with overal information about the user at registration
        df_phasestart (DataFrame): Frame the start of each program phase per user
        df_events (DataFrame): Frame with the log of each login and forum visit of the user
        df_forumpost (DataFrame): Frame with all the forum posts created by the user
        df_forumthread (DataFrame): Frame with all the forum threads created by the user
        df_thread (DataFrame): Frame with the all the threads visited by the user
        
    Returns:
        All dataframes above, in the same order with fixed column types and values                 
    """
    
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
    """ Remove the data from participants before a given date.
    
    Arguments:
        df (DataFrame): data to be cleaned
        date_columns (string): name of the date column
        date_to_filter (string): date in year-month-day, data before this date is removed
    
    Returns:
        df_filter (DataFrame): cleaned data
    """
    df_filter = df[(df[date_column] > date_to_filter)]
    #print("Unique Ids in ProgramParticipation that started after 2016 (start date of ConsumptionRegistration): ",df_filter.Id.nunique())
    return(df_filter)
    
      
def merge_multactivity_program(df_activity,df_program, leftdate_on, rightdate_on,left_by,right_by):
    """This merge is different because there are multiple forum posts and some participants have 
       multiple entries in the Program table, so we match per date as well, more can be found on 2 - Database_tables_explanation.docx
       
     Arguments:
        df_activity (DataFrame): data with the multiple occurences, e.g forum posts,logins
        df_program (DataFrame): data about the participant, start, program, etc...
        leftdate_on (string): name of the date column for the df_activity DataFrame
        rightdate_on (string): name of the date column for the df_Program DataFrame
        left_by (string): name of the id column to merge on df_activity
        right_by (string): name of the id column to merge on df_Program
    
    Returns
        df_merge_ (DataFrame): merged activity dataframe with participant information
    """
    
    df_activity = df_activity.sort_values(by=[leftdate_on])
    df_program = df_program.sort_values(by=[rightdate_on])
    
    df_program[right_by] = df_program[right_by].fillna(-1)
    df_program[right_by] = df_program[right_by].astype('int64')
    
    df_activity[left_by] = df_activity[left_by].fillna(-1)
    df_activity[left_by] = df_activity[left_by].astype('int64')
    
    # This merge is brilliant and take into account a matching by date as well (besides the common match by key)
    df_merge_ = pd.merge_asof(df_activity, df_program, left_on=leftdate_on, right_on = rightdate_on, left_by=left_by, 
                                       right_by = right_by)
    return(df_merge_)



def clean_feats(type_val, pros_cons_dic, df_feats):
    """ Splits the features and renames them.   
       
     Arguments:        
        time_hours (int): number of hours to compute features from 
        df_filter (DataFrame):  filtered data about the participant
        df_frame (DataFrame): data that needs to be filtered based on time, can be assingments, forum posts etc.
        type_frame (string): type of the frame df_frame, choice of [assignment,forum,events,thread]
    
    Returns
        df_temp (DataFrame): filtered df_frame merge with participant information and 
                            only information within the time_hours interval
       
    """
    feats = df_feats[df_feats['AnswerType']==type_val]
    feats = feats.drop(['AnswerType'],axis=1)
    feats = feats.rename(columns={"counts": pros_cons_dic[type_val]})
    return(feats)  
 
def match_ids(df_filter,col_id_filter, df_frame,col_id_frame):
    """ Match ids from 2 frames and return the intersection between them.
       
     Arguments:                
        df_filter (DataFrame):  filtered data about the participants        
        col_id_filter (str): name of the column that contains the ids from df_filter
        df_frame (DataFrame): dataframe that needs to be filtered based on the ids from df_filter
        col_id_frame (str): name of the column that contains the ids from df_frame
    
    Returns
        ids_inter (list): list of ids that are present in both dataframes and can be used further (for feature engineering for instance)    
    """    
    
    ids_filter = (set(df_filter[col_id_filter]))
    ids_using = (set(df_frame[col_id_frame]))
    ids_inter = list(ids_filter.intersection(ids_using))
    return (ids_inter)
    
def bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_merge_feats,feat_name):
    """ Perform feature engineering for 1 of the multiple assignemnts available in a given week.
        
     Arguments:
        df_merge_challenge (DataFrame):  data about all the assignments performed by the user (already filtered by time)
        df_filter (DataFrame): data about the users to be included in feature engineering, program, date of start, etc...
        df_program (DataFrame): data about all the users  (same as df_filter but not filtered)
        df_merge_feats (DataFrame): feature enginerring dataframe, new features will be added to this frame
        feat_name (string): name of the assignment from where the features will be created from
    
    Returns
        df_temp (DataFrame): Dataframe with the newly added binary features        
    """    

    #selecting only one type of assignment to use
    df_assignment = df_merge_challenge[df_merge_challenge.Name==feat_name]  
    
    #select only relevant columns, replaces missing, convert date to actual date format
    df_assignment = df_assignment[['Participation','Name','DateCompletedUtc']]
    df_assignment['Participation'] = df_assignment['Participation'].fillna(-1)
    df_assignment['Participation'] = df_assignment['Participation'].astype('int64')
    df_assignment['DateCompletedUtc'] = pd.to_datetime(df_assignment['DateCompletedUtc'])
    
    #since each assignment can be visited multiple times, this creats a many vs many merge, which is solved by merging based on the date as well.
    #Check the 2 - Database_tables_explanation.docx document for details and examples
    df_merge_assignment = merge_multactivity_program(df_assignment,df_program,'DateCompletedUtc','StartDateOfParticipation','Participation','Id')
    
    #the assingment frame now has information about the participant
    #add column with the feature name and keeps only id and feature name to be included in the feature engineering frame
    df_merge_assignment[feat_name] = (df_merge_assignment.Name==feat_name).astype('int32')
    df_merge_assignment = df_merge_assignment[['Id',feat_name]]
    
    #this groupby counts how many times the same assignment was visited by the user, so the final feature is continous (total visits)
    df_temp  = df_merge_assignment.groupby(['Id']).agg({feat_name:['sum']})#.unstack()
    df_merge_assignment = pd.DataFrame(columns = ['Id',feat_name])
    df_merge_assignment['Id'] = df_temp.index
    df_merge_assignment[feat_name] = df_temp.values
        
    #Finally, the new features are added to the feature engineering dataframe
    df_temp = pd.merge(df_merge_feats,df_merge_assignment,left_on = 'Id', right_on='Id',how = 'left')
    
    return(df_temp)
                            

def filter_time_tables(time_hours,df_filter,df_frame, type_frame, name_date):
    """ Filter activities done within a certain time (defined outside) after the start of the program.
        Add information from the participant to the filtered activity frame
       
     Arguments:        
        time_hours (int): number of hours to compute features from 
        df_filter (DataFrame):  filtered data about the participant
        df_frame (DataFrame): data that needs to be filtered based on time, can be assingments, forum posts etc.
        type_frame (string): type of the frame df_frame, choice of [assignment,forum,events,thread]
    
    Returns
        df_temp (DataFrame): filtered df_frame merge with participant information and 
                            only information within the time_hours interval
       
    """
    
    #A few relevant columns to be added to the activity frame
    df_temp = df_filter[['Id','StartDateOfParticipation','GoalOfProgram', 'Participator']]
    df_temp['Id'] = df_temp['Id'].astype('int64')
    
    #depending on the type of activity, a different merge has to be performed
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
    """  Compute features for the initial or target consumption data per day of the week
        
     Arguments:        
        df_day (DataFrame): data about the initial and target consumption of the user
        target_or_consumption (str): If the features are from the Initial consumption or the Target
        
    Returns
        df_feats (DataFrame): Dataframe with 7 features (per day of the week) of how many units were consumed or are the target 
    """
    
    #selects the initial or target consumption based on type
    df_temp = df_day[df_day['Type']==target_or_consumption]
    
    #group per day of the week and sums the consumption
    df_temp  = df_temp.groupby(['Participation','DayOfWeek']).agg({'Value':['sum']}).unstack()
    
    #new dataframe to store the new features
    df_feats = pd.DataFrame()

    #insert the new features computed in the groupby into the new frame, also used to rename the days from number do Monday, Tuesday, etc..
    df_feats['Participation'] = df_temp['Value']['sum'][0].index
    df_feats['Monday' + target_or_consumption] = df_temp['Value']['sum'][0].values
    df_feats['Tuesday' + target_or_consumption] = df_temp['Value']['sum'][1].values
    df_feats['Wednesday' + target_or_consumption] = df_temp['Value']['sum'][2].values
    df_feats['Thursday' +  target_or_consumption] = df_temp['Value']['sum'][3].values
    df_feats['Friday' + target_or_consumption] = df_temp['Value']['sum'][4].values
    df_feats['Saturday' + target_or_consumption] = df_temp['Value']['sum'][5].values
    df_feats['Sunday' + target_or_consumption] = df_temp['Value']['sum'][6].values
    
    return (df_feats)

        
def select_participants_from_phase(df_program, program, df_merge_challenge,df_phasestart):
    """ Select only data from program type==Challenge as suggested by Paul
        Select only the assignments from phase 1, since we are only using data from the first days to predict success

     Arguments:
        df_program (DataFrame):  data about the participant, start, program, etc...
        program (int): number of the program to be used, check program_dict for the values
        df_merge_challenge (DataFrame): data about the assignments performed by the users, like, reading, writing, etc...
        df_phasestart (DataFrame): data about the date the user started each phase
    
    Returns
        df_merge_challenge (DataFrame): only assingments from phase 1 of the intervention
        df_filter (DataFrame): filtered program data, only participants of type Challenge
    """
    
    #Paul suggested we only use data from challenge since it is the new program. Participants from old program dont have phase either
    df_filter = df_program[df_program.Type=='Challenge']
    df_filter = df_filter[df_filter['Program']==program]
    
    #Get the assingments from  phase 1 because we just need the features from phase 1 for now 
    #(first 2 or 3 dyas of intervention are still phase 1)
    df_merge_challenge = df_merge_challenge[df_merge_challenge['Phase']==1]
    
    #here I check the assingments from phase 1 that were finished/revisited after the start of phase 2
    #get phase 2 related information
    df_phase_filter = df_phasestart[df_phasestart.Phase==2]
    df_phase_filter = df_phase_filter.drop(['Id','Phase'],axis=1)
    
    df = pd.merge(df_merge_challenge,df_phase_filter, on='Participation',how='left')
    df['DateCompletedUtc'] = pd.to_datetime(df['DateCompletedUtc'])
    
    df['After Phase 2']= df.DateCompletedUtc>df.DateStarted
    
    df_merge_challenge = df[df['After Phase 2']==False]
    
   # print("Assignment from Program: %s selected!"%(program_dict[PROGRAM]))
    
    return (df_merge_challenge,df_filter)


def feature_engineering_pros_cons(df_filter,df_pcusing,pros_cons_dic):
    """ First step of feature engineering, adds the total of pros of stopping (long and short term)
        and the cons of continuing using (also long and short term)
        
     Arguments:
        df_filter (DataFrame):  data about the participant, start, program, etc...
        df_pcusing (DataFrame): data about the pros and cons that were written by the user
        pros_cons_dic (dict): dictionary that connects Answer type to pros or cons 
    
    Returns
        df_merge_feats (DataFrame): First features from feature engineering        
    """

    #Get the ids that are avaiable in both filtered participants and the pros and cons dataframe             
    ids_match = match_ids(df_filter,'Id', df_pcusing,'Participation')
    
    #keeps only the ids that are available in both dataframe
    df_temp = df_pcusing[df_pcusing.Participation.isin(ids_match)]
    #add column with the size of the pro or con written by the user (how long is the answer)
    df_temp['Answer Length'] = df_temp['Answer'].str.len()    
    
    #groupby to count how many items were written per answertype, so how many pros_short, cons_short, etc..
    df_feats = df_temp.groupby(['Participation', 'AnswerType']).size().reset_index(name='counts')    
    
    #here we convert the grouped frame into actual feature columns and rename them 
    pros_short = clean_feats(3,pros_cons_dic,df_feats)
    pros_long = clean_feats(4,pros_cons_dic,df_feats)
    cons_short = clean_feats(5,pros_cons_dic,df_feats)
    cons_long = clean_feats(6,pros_cons_dic,df_feats)
        
    # we merge all the separated feature frames into a single frame
    merge_feats = pd.merge(pros_short,pros_long,on='Participation',how = 'outer')
    merge_feats = pd.merge(merge_feats,cons_short,on='Participation', how= 'outer')
    merge_feats = pd.merge(merge_feats,cons_long,on='Participation',  how = 'outer')    
        
    #We select a few relevant columns about the participant to be included in the final dataset frame (df_merge_feats)
    df_merge_feats = pd.merge(df_filter[['Id','Phase','StartDateOfParticipation','GoalOfProgram']],merge_feats,left_on = 'Id', right_on='Participation',how = 'left')
    
    #df_merge_feats has the first features from feature engineering and some information about the participants
    
    return (df_merge_feats)

def feature_engineering_assignments(df_merge_challenge,df_filter,df_program,df_merge_feats):
    """ Perform feature engineering for all 4 assignments from phase 1, start video, afspraken maken, etc,
        
     Arguments:
        df_merge_challenge (DataFrame):  data about all the assignments performed by the user (already filtered by time)
        df_filter (DataFrame): data about the users to be included in feature engineering, program, date of start, etc...
        df_program (DataFrame): data about all the users  (same as df_filter but not filtered)
        df_merge_feats (DataFrame): feature enginerring dataframe, new features will be added to this frame
    
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added binary features        
    """
    
    #bin_feat_eng_phase1 is defined to deal with the multiple assignments
    df_temp = bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_merge_feats,'Start video')
    df_temp = bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_temp,'Afspraken maken')
    df_temp = bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_temp,'Voor- en nadelen')
    df_merge_feats = bin_feat_eng_phase1(df_merge_challenge,df_filter,df_program,df_temp,'Jouw afspraken')
    
    return (df_merge_feats)

def feature_engineering_assignment_agreement(df_filter,df_merge_feats,df_assign_answer,pros_cons_dic):
    """ Perform feature engineering for the assignment writing agreements with yourself 
        
     Arguments:        
        df_filter (DataFrame): data about the users to be included in feature engineering, program, date of start, etc...
        df_merge_feats (DataFrame): feature enginerring dataframe, new features will be added to this frame
        df_assign_answer (DataFrame):  data about the writing agreemewnts assignmentsperformed by the user
        pros_cons_dic (dict): dictionary that connects Answer type to pros or cons

    
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features, number of agreements and total length        
    """
    
    #Get the ids that are avaiable in both filtered participants and the pros and cons dataframe             
    ids_match = match_ids(df_filter,'Id', df_assign_answer,'Participation')
     
    #keep only the ids that are available in both dataframe
    df_temp = df_assign_answer[df_assign_answer.Participation.isin(ids_match)]
    #add column with the size of the text written by the user (how long is the answer)
    df_temp['Agreement Length'] = df_assign_answer['Agreement'].str.len()
     
    #groupby to count how many items were written per user
    df_feats = df_temp.groupby(['Participation']).size().reset_index(name='Number of Agreements')
    #groupby to sum the length of all agreements into as single total value 
    df_feats['Total Agreement Length'] = df_temp.groupby(['Participation']).agg({'Agreement Length':['sum']})['Agreement Length']['sum'].values    
    df_merge_feats = df_merge_feats.drop(['Participation'],axis=1)
    
    #here we select which columns will be included in the feature variable
    df_merge_feats = pd.merge(df_merge_feats,df_feats,left_on = 'Id', right_on='Participation',how = 'left')
    
    #in case hours was used to filter, some participants will have pros and cons but the voo-en nadelen will be set to nan, so we remove pros and cons     
    # for key in pros_cons_dic.keys():    
    #     df_merge_feats[pros_cons_dic[key]] = df_merge_feats['Voor- en nadelen']*df_merge_feats[pros_cons_dic[key]]

    
    return(df_merge_feats)


def feature_engineering_consumption_features(df_day,df_merge_feats):
    """ Perform feature engineering for the initial and target consumption data (filled in during registration)
        
     Arguments:        
        df_day (DataFrame): data about the initial and target consumption of the user
        df_merge_feats (DataFrame): feature enginerring dataframe, new features will be added to this frame      
        
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features, number of agreements and total length        
    """
            
    #compute the features per type, initial or target (how much per week day - total 7 features per frame)
    df_initial = compute_target_consumption_features(df_day,'Initial')
    df_target = compute_target_consumption_features(df_day,'Target')
     
    #combine the initial and target features
    df_feats_cons = pd.merge(df_initial, df_target,on = 'Participation')
    
    #merge with the other features and remove redundant columns    
    df_merge_feats = pd.merge(df_merge_feats,df_feats_cons, left_on= 'Id',right_on = 'Participation',how = 'left')
    df_merge_feats = df_merge_feats.drop(['Participation_x','Participation_y'],axis=1)
    df_feats_cons = df_feats_cons.drop(['Participation'],axis=1)
    
    #Finally, we fill the missing initial and target features with -1 so we know they are missing but they dont break the code
    feats = df_feats_cons.columns
    df_merge_feats[feats] = df_merge_feats[feats].fillna(-1)

    return(df_merge_feats)


def feature_engineering_forum_and_login_events(TIME_HOURS,df_filter,df_merge_feats,df_events):
    """ Perform feature engineering for the login and forum visits events
        
     Arguments:        
        TIME_HOURS (int): maximum number of hours that the features can be computed, after that it is ignored
        df_filter (DataFrame): df_filter (DataFrame): data about the users to be included in feature engineering      
        df_merge_feats (DataFrame): feature enginerring dataframe, new features will be added to this frame 
        df_events (DataFrame): frame with the registration of each login and forum visit event
        
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features, total number of logins and forum visits in the first TIME_HOURS interval       
    """
    
    #filter by time and adding df_filter columns to df_events
    df_temp = filter_time_tables(TIME_HOURS, df_filter, df_events,'events','DateOfEvent')
    
    #select only forum visit events and counting teh total numbe rusing groupby
    df_temp_visit = df_temp[df_temp['Title']=='Forum Visited']
    df_temp_visit = df_temp_visit.groupby(['Id_y']).size().reset_index(name='Number of Forum Visits')
    
    #add the new forum features to the feature engineering frame and drop repeated features
    df_merge_feats = pd.merge(df_merge_feats,df_temp_visit,left_on = 'Id', right_on='Id_y',how = 'left')
    df_merge_feats = df_merge_feats.drop(['Id_y'],axis=1)
    
    #do the same to logins
    df_temp_login = df_temp[df_temp['Title']=='User Login']
    df_temp_login = df_temp_login.groupby(['Id_y']).size().reset_index(name='Number of Logins')
    df_merge_feats = pd.merge(df_merge_feats,df_temp_login,left_on = 'Id', right_on='Id_y',how = 'left')
    df_merge_feats = df_merge_feats.drop(['Id_y'],axis=1)
    
    return(df_merge_feats)

def feature_engineering_forumpost_thread(TIME_HOURS,df_filter,df_forumpost,df_merge_feats,time_forum, var_name):
    """ Perform feature engineering for the total of forum posts or threads created
        
     Arguments:        
        TIME_HOURS (int): maximum number of hours that the features can be computed, after that it is ignored
        df_filter (DataFrame): df_filter (DataFrame): data about the users to be included in feature engineering      
        df_forumpost (DataFrame): frame with each forum post or thread created by each user
        df_merge_feats (DataFrame): feature engineering dataframe, new features will be added to this frame 
        time_forum (str): name of date column in the df_forumpost frame
        var_name (str): name of the new variable to be added to df_merge_feats
        
        
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features, total number of logins and forum visits in the first TIME_HOURS interval       
    """
    
    # filter by time and adding df_filter columns to df_forumpost, also takes cares of the multiple vs multiple merging problem
    df_temp = filter_time_tables(TIME_HOURS, df_filter, df_forumpost,'forum',time_forum)
    
    #groupby to sum all the total of forum posts per user
    df2 = df_temp.groupby(['Id_y']).size().reset_index(name=var_name)
    df2.Id_y = df2.Id_y.astype('int32')
    
    #add the total number of forum/thread posts per user as a feature to df_merge_feats and drop redundant columns
    df_merge_feats = pd.merge(df_merge_feats, df2, left_on = 'Id',right_on = 'Id_y',how = 'left')
    df_merge_feats = df_merge_feats.drop(['Id_y'],axis=1)

    return (df_merge_feats)


def add_consumption_before_last_phase(number_of_days_before, df_cons, df_phasestart,df_merge_feats):
    """ Computes how many units were consumed per day before the start of the last phase of the intervetion
        This information is addded to df_merg_feats but it is actually used for defining the label
        
     Arguments:        
        number_of_days_before (int): consumption of how many days before the start of the last phase will be included 
        df_cons (DataFrame): data about the daily consumption registered by the user
        df_phasestart (DataFrame): frame with the start of each phase per user
        df_merge_feats (DataFrame): feature engineering dataframe, new features will be added to this frame 
                
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features about daily consumption before last phase
    """

    # convert the number of days to time delta to allow data operations
    days_subtract = timedelta(days=number_of_days_before)

    df_cons6 = df_cons.copy()

    #get the last phase they achieved
    df_phasegroup = df_phasestart.sort_values('DateStarted').groupby('Participation').tail(1)[['Phase', 'DateStarted', 'Participation']]
    
    #subtract the number_of_days_before from the start of the last phase so we know when to start computing the consumption features
    df_phasegroup['date_7days'] = df_phasegroup['DateStarted'] - days_subtract

    #add information from phase to consumption
    df_merge_phasecons = pd.merge(df_cons6,df_phasegroup,on='Participation',how='left')
    
    #These participants did not reach further than phase 1
    df_merge_phasecons = df_merge_phasecons[df_merge_phasecons['date_7days'].notna()]

    #convert date colums to datetime
    df_merge_phasecons['date_7days'] = pd.to_datetime(df_merge_phasecons['date_7days'])
    df_merge_phasecons['DateOfRegistration'] = pd.to_datetime(df_merge_phasecons['DateOfRegistration'])

    #get only information from the interval between the start of the last phase and N days before
    df_merge_phasecons = df_merge_phasecons[df_merge_phasecons['DateOfRegistration']>df_merge_phasecons['date_7days']]
    df_merge_phasecons = df_merge_phasecons[df_merge_phasecons['DateOfRegistration']<df_merge_phasecons['DateStarted']]

    df = df_merge_phasecons.copy()

    #groupby to get how many units per user per day of the week were used,day_name convert date of actualy day of the week
    #we need to know which day of the week the consumption is to be able to compare to the target consumption
    df['DayOfWeek'] = df['DateOfRegistration'].dt.day_name()
    df_temp = df.groupby(['Participation','DayOfWeek']).agg({'NumberOfUnitsConsumed':['sum']}).unstack()
    df_temp = df_temp.reset_index(level=['Participation'])
    df_temp = df_temp.droplevel(1,axis=1)
    df_temp = df_temp.droplevel(0,axis=1)
    
    #renaming some columns
    cols = list(df_temp.columns)
    cols[0] = 'Participation'
    df_temp = pd.DataFrame(df_temp.values,columns = cols)

    #merge features to the others.
    df_temp  = pd.merge(df_merge_feats,df_temp,left_on ='Id',right_on = 'Participation',how='left')
    df_temp = df_temp.drop('Participation',axis=1)
    
    return(df_temp)


def feature_engineering_thread_vieweing(TIME_HOURS,df_thread,df_filter,df_merge_feats):
    """ Computes how many forum threads were viewed by the user
        
     Arguments:        
        TIME_HOURS (int): maximum number of hours that the features can be computed, after that it is ignored
        df_thread (DataFrame): data about the threads that were viewed by the user
        df_filter (DataFrame): df_filter (DataFrame): data about the users to be included in feature engineering      
        df_merge_feats (DataFrame): feature engineering dataframe, new features will be added to this frame 
                
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features about the total threads viewed
    """

    # copy the frame to manipulate without runing the data
    df_temp = df_filter.copy()
    
    # filter threads viewed later than the time frame and add information about the user to df_thread
    df_temp = filter_time_tables(TIME_HOURS, df_temp, df_thread,'thread','Date')
    
    # group by to sum the total of threads viewed 
    df_temp = df_temp.groupby(['Id_y']).size().reset_index(name='Number of Thread Views')
    
    # merge with df_merge_feats
    df_merge_feats  = pd.merge(df_merge_feats,df_temp,left_on ='Id',right_on = 'Id_y',how='left')
    
    #remove redundant columns
    df_merge_feats = df_merge_feats.drop('Id_y',axis=1)
    
    return(df_merge_feats)


def feature_engineering_badge(TIME_HOURS,df_filter,df_partbadge,df_merge_feats):
    """ Computes how many badges were earned by the user during a given time frame
        badges can be earned by doing assignments, achieving milestones, etc..
        
     Arguments:        
        TIME_HOURS (int): maximum number of hours that the features can be computed, after that it is ignored
        df_filter (DataFrame): df_filter (DataFrame): data about the users to be included in feature engineering      
        df_partbadge (DataFrame): data about the badges earned by the user       
        df_merge_feats (DataFrame): feature engineering dataframe, new features will be added to this frame 
                
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features about the total number of badges earned
    """
    # filter badges earned after the time frame and add information about the user to df_partbadge
    df_temp = filter_time_tables(TIME_HOURS, df_filter, df_partbadge,'assignment','DateAssignedUtc')
    
    # groupby to compute the total of badges earned
    df_temp = df_temp.groupby(['Participation']).size().reset_index(name='Number of Participation Badges')
    
    # add the new variable and delete redundant columns
    df_merge_feats  = pd.merge(df_merge_feats,df_temp,left_on ='Id',right_on = 'Participation',how='left')    
    df_merge_feats = df_merge_feats.drop('Participation',axis=1)

    return(df_merge_feats)


def feature_engineering_achievementlike(TIME_HOURS,df_filter,df_achievelike,df_merge_feats):
    """ Computes how many achievements were liked/viewed by the user
        
        
     Arguments:        
        TIME_HOURS (int): maximum number of hours that the features can be computed, after that it is ignored
        df_filter (DataFrame): df_filter (DataFrame): data about the users to be included in feature engineering      
        df_achievelike (DataFrame): data about the achievements the user like/viewed      
        df_merge_feats (DataFrame): feature engineering dataframe, new features will be added to this frame 
                
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features about the total number achievements the user viewed
    """
    
    # drop records with missing date
    df_achievelike = df_achievelike.dropna(subset=['DateReadUtc'])
    
    #convert date to proper format
    df_achievelike['DateReadUtc'] = pd.to_datetime(df_achievelike['DateReadUtc'])    
    
    # filter achievements viewed after the time frame and add information about the user to df_achievelike
    df_temp = filter_time_tables(TIME_HOURS,df_filter,df_achievelike, 'thread', 'DateReadUtc')
    
    df_temp = df_temp.rename(columns={'Id_y':'Id'})
    
    # groupby to compute the total of achievements viewed
    df_temp = df_temp.groupby(['Id']).size().reset_index(name='Number of Achievement Likes')
    
    # add new variable to df_merge_feats
    df_merge_feats = pd.merge(df_merge_feats,df_temp,on='Id',how = 'left')
    
    return(df_merge_feats)

        
def feature_engineering_consumption_first_days(TIME_HOURS, df_cons, df_filter,df_merge_feats):
    """ Computes how many units were consumed per day in the first few days of intervention
                
     Arguments:        
        TIME_HOURS (int): maximum number of initial hours to be used for computing the features (48,72,96)
        df_cons (DataFrame): data about the daily consumption registered by the user
        df_filter (DataFrame): data about the users
        df_merge_feats (DataFrame): feature engineering dataframe, new features will be added to this frame 
                
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features about total consumption in the first days of intervention use
    """

    # add information about the user to the consumption table, left join to keep all the consumptions
    df_merge_cons = pd.merge(df_cons,df_filter,left_on='Participation',right_on='Id',how = 'left')
    
    # drop ids that did not match with the df_filter
    df_merge_cons = df_merge_cons[~df_merge_cons.Id_y.isna()]

    # convert date columns to proper format
    df_merge_cons['DateSaved'] = pd.to_datetime(df_merge_cons['DateSaved'])
    
    # compute the different between the date the consumption was registered and the start of the participation in the program
    df_merge_cons['Diff'] = df_merge_cons['DateSaved'] - df_merge_cons['StartDateOfParticipation']
    
    # subset of important columns
    df = df_merge_cons[['Participation','DateOfRegistration','DateSaved','NumberOfUnitsConsumed','StartDateOfParticipation','Diff']]
    
    # convert the difference time variable to hours so we can compare witg time_hours
    df = convert_dttime_to_hours(df,'Diff','Total time')
    
    # keep only entries that were within the time window
    df = df[df['Total time']<TIME_HOURS]
    
    # compute total units consumed with groupby
    df_temp = df.groupby(['Participation']).agg({'NumberOfUnitsConsumed':['sum']})#.unstack()
    
    # clean the information and add merge it to the other features
    df_frame = pd.DataFrame(columns = ['Id','Total interval consumption'])
    df_frame['Id'] = df_temp.index
    df_frame['Total interval consumption'] = df_temp.values    
    
    df_merge_feats  = pd.merge(df_merge_feats,df_frame,on = 'Id',how='left')
    
    return (df_merge_feats)


def feature_engineering_diaryrecord(TIME_HOURS,df_filter,df_diary,df_merge_feats):
    """ Computes how many diary entries the user did and the total length (how much they wrote) 
                
     Arguments:        
        TIME_HOURS (int): maximum number of initial hours to be used for computing the features (48,72,96)
        df_filter (DataFrame): data about the users
        df_cons (DataFrame): data about the diary records        
        df_merge_feats (DataFrame): feature engineering dataframe, new features will be added to this frame 
                
    Returns
        df_merge_feats (DataFrame): Dataframe with the newly added features about total number of diary records
    """
    
    # filter diary enteies after the time frame and add information about the user to df_diary
    df_temp = filter_time_tables(TIME_HOURS, df_filter, df_diary,'assignment','DateCreated')

    # get the ids that match between filter and diary
    ids_match = match_ids(df_filter,'Id', df_diary,'Participation')

    # keep only ids that match
    df_temp = df_temp[df_temp.Participation.isin(ids_match)]
    
    # compute length of diary entries
    df_temp['Content Length'] = df_temp['Content'].str.len()    

    # compute total number of diary entries
    df_feats = df_temp.groupby(['Participation']).size().reset_index(name='Diary Entries')  

    # include new variables to the df_merge_feats
    df_merge_feats = pd.merge(df_merge_feats,df_feats,left_on = 'Id', right_on='Participation',how = 'left')

    return(df_merge_feats)

def filter_assigment_ids_after_phase(assign_name,df_merge_challenge,df_phasestart,df_filter):    
    """ *This function is not used in the main code, was used only for visualization and test purposes*
    Removes all assignments performed by the user after a given phase started
    
    Arguments:
        assign_name (list): list of strings with the names of the assignments to be filtered
        df_merge_challenge (DataFrame): data with the assignments from the users
        df_phasestart (DataFrame): data of the date each participants started each phase
        df_filter (DataFrame): data of some basic user information
    Returns:
        df_merge_challenge (DataFrame): dataframe with filtered assignments             
        df_filter (DataFrame): data of basic info without the extra information
    """
        
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


def create_features_time(df,feat_name,time_name,new_feat_name):
    df_feat = df[df['Name']==feat_name][['Participation',time_name]]
    df_feat = convert_dttime_to_hours(df_feat,time_name,new_feat_name)
    df_feat = df_feat.rename(columns={'Participation':'Id'})
    df_feat = df_feat.drop(time_name,axis=1)
    return (df_feat)

def feature_engineering_time_assigment(df_merge_challenge_feat,df_merge_feats,df_filter,date_start,date_finished):
    """ *This function is not used in the main code, since the interpretation was a bit confusing*
    Computes from the start  to the moment the user saved the assignments, how long they took to do it
    This IS NOT the time spent in a given assignment
    
    Arguments:        
        df_merge_challenge_feat (DataFrame): data with the assignments from the users
        df_merge_feats (DataFrame):  Dataframe with the features previously computed
        df_filter (DataFrame): data of some basic user information
        date_start (string): Name of column of the start of the assingment
        date_finished (string): Name of column of when the user saved the assingment
    Returns:
        df_merge_feats (DataFrame): Dataframe with the newly added features about total time until saving an assignment
    """

    df_merge_challenge_feat = pd.merge(df_merge_challenge_feat,df_filter[['Id',date_start]],left_on = 'Participation', right_on='Id')
    df_merge_challenge_feat['Complete Time'] = df_merge_challenge_feat[date_finished] - df_merge_challenge_feat[date_start]
    #df_merge_challenge_feat = df_merge_challenge_feat.drop(['Id_x','Id_y'],axis=1)
    
    df = df_merge_challenge_feat.groupby(['Participation','Name']).first()
    #df['Name'] = df.index.get_level_values(1)
    df = df.reset_index(level=['Participation'])
    df = df.reset_index(level=['Name'])
    
    #Create features per type of assigment
    df_video = create_features_time(df,'Start video','Complete Time','StartVideo_time')
    df_voorennadelen = create_features_time(df,'Afspraken maken','Complete Time','AfsprakenMaken_time')
    df_afspraak = create_features_time(df,'Voor- en nadelen','Complete Time','ProsCons_time')
    df_jouw= create_features_time(df,'Jouw afspraken','Complete Time','Jouw_afspraken_time')    
    
    df_merge_feats = pd.merge(df_merge_feats,df_video,on='Id',how='left')
    df_merge_feats = pd.merge(df_merge_feats,df_voorennadelen,on = 'Id',how='left')
    df_merge_feats = pd.merge(df_merge_feats,df_afspraak,on = 'Id',how='left')
    df_merge_feats = pd.merge(df_merge_feats,df_jouw,on = 'Id',how='left')
    return(df_merge_feats)


def remove_user_future_edits(df_merge_challenge,df_filter,df_merge_feats,time_hours):
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