import csv

with open('ltrain.csv', 'r') as f:
    lrows = [row for row in csv.reader(f, delimiter=',')]

with open('ctrain.csv', 'r') as f:
    crows = [row for row in csv.reader(f, delimiter=',')]

cdict = {}
for i, crow in enumerate(crows):
    cdict[crow[0]] = i

nrows = []
for i, lrow in enumerate(lrows):
    if lrow[0] not in cdict:
        continue
    crow = crows[cdict[lrow[0]]]
    nrow = lrow[:4] + crow[2:6] + lrow[4:] + crow[7:]
    nrows.append(nrow)

with open('train.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    for row in nrows:
        writer.writerow(row)



with open('lvalid.csv', 'r') as f:
    lrows = [row for row in csv.reader(f, delimiter=',')]

with open('cvalid.csv', 'r') as f:
    crows = [row for row in csv.reader(f, delimiter=',')]

cdict = {}
for i, crow in enumerate(crows):
    cdict[crow[0]] = i

nrows = []
for i, lrow in enumerate(lrows):
    if lrow[0] not in cdict:
        continue
    crow = crows[cdict[lrow[0]]]
    nrow = lrow[:4] + crow[2:6] + lrow[4:] + crow[7:]
    nrows.append(nrow)

with open('valid.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    for row in nrows:
        writer.writerow(row)
