# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:26:31 2020

@author: laramos
"""
import sqlite3

fd = open(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\LucasRamosData\dbo.Achievement.Table.sql", 'r')
sqlFile = fd.read()
fd.close()


fd = open(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\LucasRamosData\dbo.ForumPostLike.Table.sql", 'r')
sqlFile = fd.read()
fd.close()






import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    create_connection(r"pythonsqlite.db")
    
    
    
import sqlite3
from sqlite3 import OperationalError

conn = sqlite3.connect('pythonsqlite.db')
c = conn.cursor()

# Open and read the file as a single buffer
fd = open(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\LucasRamosData\dbo.ConsumptionDetailItem.Table.sql", 'rb')
sqlFile = fd.read()
fd.close()

# all SQL commands (split on ';')
sqlCommands = sqlFile.split(';')

# Execute every command from the input file
for command in sqlCommands:
    # This will skip and report errors
    # For example, if the tables do not yet exist, this will skip over
    # the DROP TABLE commands
    try:
        c.execute(command)
    except OperationalError:
        print("Command skipped: ")
        
        
        
import pandas as pd
import re
from datetime import datetime
#df = pd.read_excel(r"\\amc.intra\users\L\laramos\home\Desktop\test.xls")   

#df = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\test.txt", sep=",",encoding = "ISO-8859-1")
    
file1 = open(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\dbo.ForumThread.txt", 'r') 
lines = file1.readlines()     

error = list()
line_error = list()
new_lines = list()

for i,line in enumerate(lines): 
    line = line.replace('\x00','')
    if line.count(',')>7:
        print("Line:", i) 
        error.append(line)
        line_error.append(i)
    new_lines.append(line)
    #break

fixed = list()

for i,s in enumerate(error):
    s1 = s.replace('[\x00,\x00 ','\x00')
    s = s1.replace('\x00','')
    #s = s.replace('[a-z][,]','')
        
    pattern = '[a-z][,][ ][a-z]|[\D][,][ ][\D]|[0-9][,][0-9][\D]||[a-z][,][a-z]'
    
    #pat_id = re.search(pattern,s)
    pat_id = re.findall(pattern,s)
    txt = pat_id[0]
    txt_replace = txt.replace(',',' ')
    s = s.replace(txt,txt_replace)
    if s.count(',')>7:
        print(s)
        parts = s.split(',')
        date = parts[2]
        try:
            datetime_object = datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
        except:
            not_found = True
            i = 3
            while not_found and i<len(parts):
            #for i in range(3,len(parts)):
                try:            
                    date = parts[i]
                    print(date)
                    datetime_object = datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
                    not_found = False    
                except:            
                    i = i+1
                    
        new_str = parts[0]+','          
        for  j in range(1,len(parts)):
            if j<i-1:
                new_str = new_str + parts[j]
            else:
                new_str = new_str + parts[j]+','
                
        if new_str.count(',')>7:
            new_str = new_str[0:len(new_str)-1]
            if new_str.count(',')>7:
                print("WRONG!!!",new_str)
            else:
                if new_str.count(',')<7:
                    print("WRONG2!!!",new_str)
                
        fixed.append(new_str)
    

for i in range(0,len(fixed)):
    new_lines[line_error[i]] = fixed[i]


with open(r'\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\listfile.txt', 'wb') as filehandle:
    for listitem in new_lines:
        filehandle.write('%s' % str(listitem))
    filehandle.close()



#pat_id = re.search('\x00,[\x005-\x009]',s)
pat_id = re.search('[a-z][,][ ][a-z]',s)
print(pat_id)
txt = pat_id.group(0)
txt_replace = txt.replace(',',' ')
s = s.replace(txt,txt_replace)

s.replace('\x00','')

s = s.replace(' ','')


pattern = '\w\w[w]'
re.findall(pattern, 'hi how are you &s')


pattern = '[a-z][,][ ][a-z]|[\D][,][ ][\D]'
re.search(pattern,s).group(0)



s = lines[188]

s = s.replace('\x00','')

print(s)



file2 = open(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\listfile.txt", 'r') 
lines = file1.readlines() 

with open('output.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(new_lines)




#this is for fixing the dairy data
# new code
import os
import glob
import pandas as pd

#file_paths = glob.glob(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_quotes_as_separator\*.txt")
file_paths = [r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\test_enconde_2.txt"]

file_number = 0

#file1 = open(file_paths[file_number], 'r')  encoding="utf8"
file1 = open(file_paths[file_number], 'r',encoding=" Latin-1")  
path_data, name = os.path.split(file_paths[file_number])
name = name[0:len(name)-3]+'csv'
lines = file1.readlines()     
file1.close()

new_lines = list()
for i,s in enumerate(lines):
    new_lines.append(s.replace('\x00',''))

n_cols = new_lines[0].count('","')   

cols = new_lines[0]
cols = cols.replace('ÿþ','')
cols = cols.replace('\n','')


cols = cols.split('","')
cols[0] = cols[0].replace('"','')
cols[n_cols] = cols[n_cols].replace('"','')

k = 2

line = new_lines[k].replace('\n','')
line = new_lines[k].split('","')
print(line)
line[0] = line[0].replace('"','')
line[n_cols] = line[n_cols].replace('"\n','')
df = pd.DataFrame([line], columns = cols)

i = 4

error = list()
line_error = list()

error_next = list()

while i<(len(new_lines)-1):
#while i<200:
    if "8706" in (new_lines[i]):
        print(i)
        #error.append(new_lines[i])
        #error_next.append(new_lines[i+1])
    if (new_lines[i].count('","'))!=len(cols)-1:
        if (new_lines[i].count('","'))>0:
             line = new_lines[i]+new_lines[i+1]
             line = line.replace('\n','')
             line = line.split('","')
             if len(line)==len(cols):
                 line[n_cols] = line[n_cols].replace('"\n','')
                 df2 = pd.DataFrame([line], columns = cols)        
                 df = pd.concat([df, df2])
                 i=i+1
             elif len(line)==len(cols)-1:
                line.append('","\n')
                line[n_cols] = line[n_cols].replace('"\n','')
                df2 = pd.DataFrame([line], columns = cols)        
                df = pd.concat([df, df2])
             else:
                error.append(line)
                line_error.append(i)
    elif (new_lines[i].count('","'))==len(cols)-1:
        line = new_lines[i].replace('\n','')
        line = line.split('","')
        line[0] = line[0].replace('"','')
        line[n_cols] = line[n_cols].replace('"\n','')
        df2 = pd.DataFrame([line], columns = cols)        
        df = pd.concat([df, df2])        
    elif new_lines[i].count('","')==0:
        j = df.shape[0]-1
        line = new_lines[i].replace('\n','')
        line = line.split('","')
        df.Content.iloc[j] = df.Content.iloc[j]+line[0]
    else:        
        error.append(new_lines[i])
        line_error.append(i)
    i = i + 1

df.to_csv(os.path.join(path_data,'DiaryRecord.csv'))

d = df.Participation.value_counts()

df1 = pd.DataFrame()
df1['part'] = np.array(d.index.values,dtype='int64')
df1['vals'] = d.values

df2 = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\query.csv",header=None)

df1 = df1.sort_values(by=['part'])
df2 = df2.sort_values(by=[0])

df3 = df1['vals'].values==df2[1].values
np.where(df3==False)

df1.part.iloc[2921]
df1.vals.iloc[2921]
df2[0].iloc[2921]
df2[1].iloc[2921]


int1 = set_events.intersection(set_prog_id)
diff1 = set_events.difference(set_prog_id)



d = df[df.Participation=="8706"]


#This is the standard code
# new code
import os
import glob
import pandas as pd

#file_paths = glob.glob(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_quotes_as_separator\*.txt")

#file_paths = [r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python\DiaryRecord.txt"]
file_paths = [r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\forum.txt"]


file_number = 0

file1 = open(file_paths[file_number], 'r')
#file1 = open(file_paths[file_number], 'r', encoding="utf8")
#file1 = open(file_paths[file_number], 'r',encoding=" Latin-1")  
path_data, name = os.path.split(file_paths[file_number])
name = name[0:len(name)-3]+'csv'
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

k = 1

line = new_lines[k].replace('\n','')
line = new_lines[k].split('","')
print(line)
line[0] = line[0].replace('"','')
line[n_cols] = line[n_cols].replace('"\n','')
df = pd.DataFrame([line], columns = cols)

i = 2

error = list()
line_error = list()

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

        error.append(new_lines[i])
        line_error.append(i)
    i = i + 2

df.to_csv(os.path.join(path_data,name))


df2 = df.drop(['AuthenticationTokenHash','AuthenticationTokenValidUntil','Avatar','TwoFactorAuthenticationCode'],axis=1)
df2.to_csv(os.path.join(path_data,name))



df = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_quotes_as_separator\dbo.ProgramParticipation.csv")

#df = df[['Id', 'InitialConsumptionOnMonday','InitialConsumptionOnTuesday']]
df2 = df.drop(['Unnamed: 0','Id'],axis=1)

df2.to_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_quotes_as_separator\testdbo.ProgramParticipation.csv",index=False)




#This is for the challenge file, it has too much text and html, so it does not fit the code above
#the informationm spreads accross multiple lines

i = 2
line = new_lines[i].replace('\n','')
c_line = line
i = 4

error = list()


while '{' not in new_lines[i]:
    line = new_lines[i].replace('\n','')
    c_line = c_line + line
    i = i + 2

line = c_line.split('","')
line[0] = line[0].replace('"','')
line[n_cols] = line[n_cols].replace('"\n','')
print(line)

df = pd.DataFrame([line], columns = cols)

while i<(len(new_lines)-1):
    line = new_lines[i].replace('\n','')
    c_line = line
    i = i + 2
    while '{' not in new_lines[i] and i<(len(new_lines)-1):
        line = new_lines[i].replace('\n','')
        c_line = c_line + line
        i = i + 2
    c_line = error[2]+error[3]
    line = c_line.split('","')
    line[0] = line[0].replace('"','')
    line[n_cols] = line[n_cols].replace('"\n','')
    df2 = pd.DataFrame([line], columns = cols)
    df = pd.concat([df, df2])  
    

df = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_quotes_as_separator\dbo.Badge.csv")

df = df.drop(['Introduction','Body','VideoUrl','IconClass','BackgroundClass','Title','Conclusion'],axis=1)
df = df.drop(['Unnamed: 0'],axis=1)
df.to_csv(os.path.join(path_data,name),index=False)
df.to_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_quotes_as_separator\dbo.Badge.csv",index=False)



    c_line = error[20]+error[21]+error[19]+error[10]
    line = c_line.split('","')
    print(len(line))
    line[0] = line[0].replace('"','')
    line[n_cols] = line[n_cols].replace('"\n','')
    df2 = pd.DataFrame([line], columns = cols)
    df = pd.concat([df, df2]) 
    
    
    
#for the text from forum

import os
import glob
import pandas as pd

#file_paths = glob.glob(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_quotes_as_separator\*.txt")

#file_paths = [r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python\DiaryRecord.txt"]
file_paths = [r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\forum.txt"]


file_number = 0

file1 = open(file_paths[file_number], 'r')
#file1 = open(file_paths[file_number], 'r', encoding="utf8")
#file1 = open(file_paths[file_number], 'r',encoding=" Latin-1")  
path_data, name = os.path.split(file_paths[file_number])
name = name[0:len(name)-3]+'csv'
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

k = 1
error = list()
line_error = list()

line = new_lines[k]#.replace('"\n','')
line_split = line.split('","')

while len(line_split)!=len(cols):
    k = k + 1 
    line2 = new_lines[k]#.replace('"\n','')
    line = line + line2
    line_split = line.split('","')
    #print(len(line_split))
    
k = k + 1    
line_split[0] = line_split[0].replace('"','')
line_split[n_cols] = line_split[n_cols].replace('"\n','')
df = pd.DataFrame([line_split], columns = cols)

while k<(len(new_lines)-1):
    
    line = new_lines[k]#.replace('"\n','')
    line_split = line.split('","')
    
    while len(line_split)!=len(cols):
        k = k + 1 
        line2 = new_lines[k]#.replace('"\n','')
        line = line + line2
        line_split = line.split('","')
        #print(len(line_split))
        
    k = k + 1    
    line_split[0] = line_split[0].replace('"','')
    line_split[n_cols] = line_split[n_cols].replace('"\n','')
    df2 = pd.DataFrame([line_split], columns = cols)        
    df = pd.concat([df, df2])    
    print(df.shape)


i = 2

error = list()
line_error = list()

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

        error.append(new_lines[i])
        line_error.append(i)
    i = i + 2

df.to_csv(os.path.join(path_data,name))


    


