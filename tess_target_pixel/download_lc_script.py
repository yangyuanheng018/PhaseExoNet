import numpy as np
import os
import pandas as pd

## read yuliang tces 
f = open('../target_info/tces.csv')
tces_data = f.readlines()
f.close()
del tces_data[0]
yuliang_ticids = []
for row in tces_data:
    ticid = row.split(',')[1]
    yuliang_ticids.append(int(ticid))
yuliang_ticids = set(yuliang_ticids)

## create a script for download
for sector in range(1, 6):
    idx = 0
    num = 0
    outfile = open('tces_sector{}.sh'.format(sector), 'w')
    outfile.write('#!/bin/sh\n')
    f = open('sectors_lc_sh/tesscurl_sector_{}_lc.sh'.format(sector))
    data = f.readlines()
    f.close()
    del data[0]
    for l in data:
        ticid = l.split('-')[6]
        if int(ticid) in yuliang_ticids:
            outfile.write(l)
            idx += 1
            if idx % 100 == 0:
               print(idx)
    print(idx)
    outfile.close()

## the number of ticids that can be discovered
lc_ticids = []
for sector in range(1, 6):
    f = open('sectors_lc_sh/tesscurl_sector_{}_lc.sh'.format(sector))
    data = f.readlines()
    f.close()
    del data[0]
    for l in data:
        lc_ticids.append(int(l.split('-')[6]))

lc_ticids = set(lc_ticids)
common = lc_ticids & yuliang_ticids
print(len(common))
