# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:42:53 2020

@author: laramos
"""

#%% Computing some descriptive statistiscs about the data
import xlwt

programs_dict = {0:'All',1:'Alcohol',2: 'Cannabis', 3: 'Gamble', 4: 'Cocaine',5: 'Smoke'}

descript_dict = {1:'Total Participants' , 2: 'Goal: Stop',3: 'Goal: Reduce',4: 'Goal: Slowly Stop',5: 'Goal: Slowly Reduce',
                 6: 'Has Phase', 7: 'Reached last phase',8: 'Duration until phase 6 (median-IQR)',9:'Has end date',10: 'Duration end date',
                 11: 'Reach Phase 2', 12: 'Reach Phase 3', 13: 'Reach Phase 4', 14: 'Reach Phase 5',15: 'Reach Phase 6',
                 16: 'Used for 1 day', 17: 'Used for 2 days', 18: 'Used for 3 days', 19: 'Used for 4 days', 20: 'Used for 5 days',
                 21: 'Used for 6 days', 22: 'Used for 1 week', 23: 'Used for 2 weeks',24: 'Used for 3 weeks',25: 'Used for 4 weeks',
                 26: 'Used for 5 weeks',27: 'Used for 6 weeks',28: 'Age', 29: 'Phase 6 in 6 weeks', 30: 'Phase 6 in 8 weeks',31: 'Phase 6 in 10 weeks',
                 32: 'Phase 6 in 12 weeks', 33: 'Phase 6 in 14 weeks',34: 'Phase 6 in 16 weeks',35: 'Is Available for Research'}

div_total = df_program.shape[0]

book = xlwt.Workbook(encoding="utf-8")    
sheet1 = book.add_sheet("Sheet 1")

#writing feature lines
for i in range(1,7):
    sheet1.write(0,i,programs_dict[i-1]) 
    
for i in range(1,len(descript_dict)+1):
    sheet1.write(i,0,descript_dict[i])     

#Write From Total Participants to Goals
for i in range(0,6):
    
    df_prog,df_prog_stop,df_prog_red,df_prog_s_stop,df_prop_s_red = filter_program_and_goal(df_program, i) 
        
    sheet1.write(1,i+1,str("%0.0f (%0.2f)"%(df_prog.shape[0],(df_prog.shape[0]*100)/df_prog.shape[0])))    
    sheet1.write(2,i+1,str("%0.0f (%0.2f)"%(df_prog_stop.shape[0],(df_prog_stop.shape[0]*100)/df_prog.shape[0])))    
    sheet1.write(3,i+1,str("%0.0f (%0.2f)"%(df_prog_red.shape[0],(df_prog_red.shape[0]*100)/df_prog.shape[0])))   
    sheet1.write(4,i+1,str("%0.0f (%0.2f)"%(df_prog_s_stop.shape[0],(df_prog_s_stop.shape[0]*100)/df_prog.shape[0])))    
    sheet1.write(5,i+1,str("%0.0f (%0.2f)"%(df_prop_s_red.shape[0],(df_prop_s_red.shape[0]*100)/df_prog.shape[0])))    


# Writes from has phase to duration and date

# Participation from PhaseStart connects to Id from ProgramParticipation

for i in range(0,6):
    
    if i==0:
        df_prog = df_program       
        df_phase_6 = df_phasestart[df_phasestart.Phase==6]
        df_prog_6 = df_prog[df_prog.Phase==6]         
        df_has_end = df_prog[~df_prog.EndDateOfParticipation.isna()]
        df_has_phase = df_prog[~df_prog.Phase.isna()]        
        div = df_prog_6.shape[0]
        div_end = df_has_end.shape[0]
        div_phase = df_has_phase.shape[0]
    else:
        df_prog = df_program[df_program['Program']==i]        
        df_phase_6 = df_phasestart[df_phasestart.Phase==6]
        df_prog_6 = df_prog[df_prog.Phase==6]
        df_has_end = df_prog[~df_prog.EndDateOfParticipation.isna()]
        df_has_phase = df_prog[~df_prog.Phase.isna()] 
        
    
        
    df_merge_phase = pd.merge(df_prog_6,df_phase_6,left_on='Id',right_on='Participation')
    duration = (df_merge_phase.DateStarted - df_merge_phase.StartDateOfParticipation).dt.days
    duration_end = (df_has_end.EndDateOfParticipation - df_has_end.StartDateOfParticipation).dt.days
    iqrh = np.nanpercentile(duration, 75, interpolation='higher')
    iqrl = np.nanpercentile(duration, 25, interpolation='lower')
    iqrh_end = np.nanpercentile(duration_end, 75, interpolation='higher')
    iqrl_end = np.nanpercentile(duration_end, 25, interpolation='lower')
    
    sheet1.write(6,i+1,str("%0.0f (%0.2f)"%(df_has_phase.shape[0],(df_has_phase.shape[0]*100)/div_phase)))    
    sheet1.write(7,i+1,str("%0.0f (%0.2f)"%(df_merge_phase.shape[0],(df_merge_phase.shape[0]*100)/div)))    
    sheet1.write(8,i+1,str("%0.0f (%0.2f - %0.2f)"%(np.median(duration),iqrl,iqrh)))          
    sheet1.write(9,i+1,str("%0.0f (%0.2f)"%(df_has_end.shape[0],(df_has_end.shape[0]*100)/div_end)))    
    sheet1.write(10,i+1,str("%0.0f (%0.2f - %0.2f)"%(np.median(duration_end),iqrl_end,iqrh_end)))          


# Writes from reach phase 2 to phase 6

row = 9 # 9 because j starts at 2 = 11
for i in range(0,6):
    if i==0:
        df_prog = df_program 
        div_phase = df_prog.shape[0]
    else:
        df_prog = df_program[df_program['Program']==i]
        div_phase = df_prog.shape[0]
    for j in range(2,7):
        df_each_phase = df_phasestart[df_phasestart.Phase==j]
        df_merge_phase = pd.merge(df_each_phase,df_prog,left_on='Participation',right_on='Id')    
        sheet1.write(row+j,i+1,str("%0.0f (%0.2f)"%(df_merge_phase.shape[0],(df_merge_phase.shape[0]*100)/div_phase)))    
        


#Write for how long the participants used, from 1 day to multiple weeks
df_diffs = df_merge_events.groupby(['EventGenerator','Program']).agg({'DateOfEvent':['first','last']})
df_diffs.columns = df_diffs.columns.map('_'.join)
#df_diffs['Days'] = df_diffs.pop('DateOfEvent_last') - df_diffs.pop('DateOfEvent_first') 
df_diffs['Start'] = df_diffs.pop('DateOfEvent_first')
df_diffs['End'] = df_diffs.pop('DateOfEvent_last')
df_diffs['Days'] = df_diffs['End'] - df_diffs['Start']
df_diffs = df_diffs.reset_index()



row = 16
time_intervals = [1,2,3,4,5,6,7,14,21,28,35,42]

for i in range(0,6):
    if i==0:
        df_prog = df_diffs 
        div_phase = df_prog.shape[0]
    else:
        df_prog = df_diffs[df_diffs['Program']==i]
        div_phase = df_prog.shape[0]
        
    for j in range(0,len(time_intervals)):
        df_prog_time = df_prog[df_prog.Days.dt.days>=time_intervals[j]]
        sheet1.write(row+j,i+1,str("%0.0f (%0.2f)"%(df_prog_time.shape[0],(df_prog_time.shape[0]*100)/div_phase)))    
               
    
row = row+len(time_intervals)

df_merge_part['YearOfBirth'] = df_merge_part['YearOfBirth'].replace([-1],np.nan)
df_merge_part['DateCreated'] =  pd.to_datetime(df_merge_part['DateCreated'])
df_merge_part['Age'] = df_merge_part['DateCreated'].dt.year - df_merge_part['YearOfBirth']
df_merge_part['Male'] = df_merge_part['Gender']==1

for i in range(0,6):

    if i==0:
        df_prog = df_merge_part 
        div_phase = df_prog.shape[0]
    else:
        df_prog = df_merge_part[df_merge_part['Program']==i]
        div_phase = df_prog.shape[0]
        
    sheet1.write(row,i+1,str("%0.0f STD(%0.2f)"%(np.nanmean(df_prog.Age),np.nanstd(df_prog.Age))))    
    #writing the missing together is too much
    #sheet1.write(row,i+1,str("%0.0f STD(%0.2f)-M(%0.2f)"%(np.nanmean(df_prog.Age),np.nanstd(df_prog.Age),np.count_nonzero(np.isnan(df_prog.Age)))))    
    

# Write the interval of time people took to reach phase 6
row = 29

weeks_to_days = [7*6, 7*8, 7*10, 7*12, 7*14, 7*16] 
for k in range(0,6):
    if k==0:
        df_prog = df_program 
        df_merge_part_prog = df_merge_part
    else:
         df_prog = df_program[df_program.Program==k]
         df_merge_part_prog = df_merge_part[df_merge_part.Program==k]
    df_prog_6 = df_prog[df_prog.Phase==6]         
    df_phase_6 = df_phasestart[df_phasestart.Phase==6]
    df_merge_phase = pd.merge(df_prog_6,df_phase_6,left_on='Id',right_on='Participation')
    df_merge_phase['Duration'] = (df_merge_phase.DateStarted - df_merge_phase.StartDateOfParticipation).dt.days 

    for i in range(0,len(weeks_to_days)):
       df_weeks = df_merge_phase[df_merge_phase['Duration'] <= weeks_to_days[i]]
       sheet1.write(row+i,k+1,str("%0.0f (%0.2f)"%(df_weeks.shape[0],(df_weeks.shape[0]*100)/df_merge_phase.shape[0])))  
       
    sheet1.write(35,k+1,str("%0.0f (%0.2f)"%(np.sum(df_merge_part_prog['IsAvailableForResearch']),(np.sum(df_merge_part_prog['IsAvailableForResearch'])*100)
                                           /df_merge_part_prog.shape[0])))  
#sheet1.write(35,2,str("%0.0f "%(np.sum(df_merge_part['IsAvailableForResearch']))))  

row = 36

#first selects all the challengs from a given phase

for j in range(1,7):
    df_each_phase = df_merge_challenge[df_merge_challenge.Phase==j]   
    names = list(df_each_phase.Name.value_counts().index)
    
    for k,nam in enumerate(names):
        
        sheet1.write(row+k,0,str("Phase %0.0f: %s"%(j,nam)))
        df_each_phase_nam = df_each_phase[df_each_phase['Name']==nam]
                
        for i in range(0,6):
            
            if i==0:
                df_prog = df_each_phase_nam 
                div_phase = df_prog.shape[0]
            else:
                df_prog = df_each_phase_nam[df_each_phase_nam['Program']==i]
                div_phase = df_prog.shape[0]
 
            sheet1.write(row+k,i+1,str("%0.0f "%(df_prog.Participation.nunique())))  
    row = row + k+1
    


book.save(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Descriptive_2.xls")  

#old inneficient code

#df_diffs = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python\df_diff_vis.csv")
# #This is terribly slow, maybe I can improve?
# ids = list(set(df_events.EventGenerator))
# df_diffs = pd.DataFrame(columns=['Id','Days','Program','Start','End'])
# df_merge_events['DateOfEvent'] = pd.to_datetime(df_merge_events['DateOfEvent'])
# for i in range(1,len(ids)):
#     df = df_merge_events[df_merge_events.EventGenerator==ids[i]]
#     df = df.sort_values(by='DateOfEvent')
#     #diffs.append((df.DateOfEvent.iloc[df.shape[0]-1] - df.DateOfEvent.iloc[0]).days)
#     df_temp = pd.DataFrame(columns=['Id','Days','Program','Start','End'])
#     if df.shape[0]>0:
#         df_diffs = df_diffs.append({'Id':ids[i], 'Days':(df.DateOfEvent.iloc[df.shape[0]-1] - df.DateOfEvent.iloc[0]).days,'Program':df.Program.iloc[0],
#                                     'Start':(df.DateOfEvent.iloc[0]).day,'End':(df.DateOfEvent.iloc[df.shape[0]-1]).day},ignore_index=True)
# df_diffs.to_csv(r'\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python\df_diff_vis_2016.csv',index=False)
