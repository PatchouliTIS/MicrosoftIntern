import win32com.client as win32
import pandas as pd
import math
import datetime
 
outlook = win32.Dispatch('outlook.application')
mail = outlook.CreateItem(0)
mail.To = 'cosmos_team_usage@microsoft.com'
mail.Subject = 'Cosmos Team Usage Last Week'

data = []
input_f = open("dist/results.tsv")
for line in input_f.readlines():
    line = line.strip().split("\t")
    table_row = {'Team' : line[0], 'Active Users' : line[4], 'Job Count' : line[1],  'Token Days' : int(float(line[3])), ' Job Count/User' : math.floor( int(line[1]) / max(int(line[4]), 1)), 'Token days/User' : math.floor(float(line[3]) / max(int(line[4]), 1))}
    data.append(table_row)
data.sort(key=lambda row: (row['Token Days']), reverse=True)
table = pd.DataFrame(data)
date01 = datetime.date.today()
mail.HTMLBody = '<html><body>' + "</br> This is the cosmos team usage of last 7 days calculated on " + str(date01.month) + "/" + str(date01.day) + table.to_html() + '</body></html>'

mail.display()
mail.Send()