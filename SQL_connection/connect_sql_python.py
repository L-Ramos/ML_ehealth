# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:35:07 2021

@author: laramos
"""
import pyodbc 
import pandas
import os

#where to save the .csv files
path_save = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\test"
#name of the table from the database
#table_name = 'ForumPost'
all_names = pandas.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\BACKUP\Exported_data\all_table_names.csv")

for table_name in all_names['names']:
    
    #change server= to the number of your server (can be found in server properties)
    #change to Database= name of the database
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=16-006877;'
                          'Database=New_Jellinek;'
                          'Trusted_Connection=yes;')
    
    ###To visualize some of the data you can use the code below
    # cursor = conn.cursor()
    # cursor.execute('SELECT * FROM New_Jellinek.dbo.ForumPost')
    # for row in cursor:
    #     print(row)
        
        
    sql = 'SELECT * FROM New_Jellinek.dbo.'+table_name
    #converts to a dataframe format
    data = pandas.read_sql(sql,conn)
    
    data.to_csv(os.path.join(path_save,table_name+'.csv'))