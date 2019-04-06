import csv
import re

with open('TransactionHistory.csv') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        label = row[5]
        label = re.sub('^V\d{4} \d{2}/\d{2} ', '', label)
        label = re.sub(' \d+$', '', label)
        print(f'{row[1]},{label}')