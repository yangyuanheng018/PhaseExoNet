import numpy as np
import os
import pandas as pd

## read tois csv
toicsv = pd.read_csv('../target_info/tois.csv', header=0, skiprows=4)

## read yuliang tces 
f = open('../target_info/tces.csv')
tces_data = f.readlines()
f.close()
del tces_data[0]
yuliang_tces = []
for row in tces_data:
    ticid = row.split(',')[1]
    yuliang_tces.append(int(ticid))
yuliang_tces = set(yuliang_tces)

## create a script for download
outfile = open('tois.sh', 'w')
idx = 0
num = 0
for sector in range(1, 33):
    f = open('sectors_lc_sh/tesscurl_sector_{}_lc.sh'.format(sector))
    data = f.readlines()
    f.close()
    del data[0]
    ind_tois = np.where(toicsv['Sectors'].str[0:2].astype(int) == sector)
    tois_ticids = np.asarray(toicsv['TIC'])[ind_tois]
    for l in data:
        ticid = l.split('-')[6]
        if int(ticid) in tois_ticids and int(ticid) not in yuliang_tces:
            outfile.write(l)
            idx += 1
            if idx % 100 == 0:
               print(idx)
print(idx)
outfile.close()

print((toicsv['Source Pipeline'] == 'qlp').sum())

## the number of tois that can be discovered.
lc_ticid = []
for sector in range(1, 33):
    f = open('sectors_lc_sh/tesscurl_sector_{}_lc.sh'.format(sector))
    data = f.readlines()
    f.close()
    del data[0]
    for l in data:
        lc_ticid.append(int(l.split('-')[6]))
lc_ticid = set(lc_ticid)

toicsv_ticid = set(toicsv['TIC'])

common = lc_ticid & toicsv_ticid
print(len(common))

