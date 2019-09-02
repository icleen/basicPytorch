import csv

with open('box_train.csv', 'r') as f:
    rows = [row for row in csv.reader(f, delimiter=',')]

rows = [[row[0], '0'] + row[1:5] + ['1'] + row[5:] for row in rows]

with open('ntrain.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    for row in rows:
        writer.writerow(row)

with open('box_valid.csv', 'r') as f:
    rows = [row for row in csv.reader(f, delimiter=',')]

rows = [[row[0], '0'] + row[1:5] + ['1'] + row[5:] for row in rows]

with open('nvalid.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    for row in rows:
        writer.writerow(row)
