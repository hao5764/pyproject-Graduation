import time
import random

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest


def loadTraindata(str):
    f = open(str, encoding='utf-8')
    sentimentlist = []
    for line in f:
        s = line.replace('?', 'NaN').strip().split(',')
        sentimentlist.append(s)
    f.close()
    dt = np.array(sentimentlist)
    dt[dt == 'noband'] = 1
    dt[dt == 'band'] = 0
    return dt


# 孤独森林对数据进行划分
def divDataSet(data):

    X_cols = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
              "x16", "x17", "x18", "x19"]

    it = IsolationForest(contamination=0.1)

    iForest = it.fit(data[:])

    pred = it.predict(data[:])
    print(data.shape)
    print(pred.shape)
    #np.insert(data,21,values=pred,axis=1)

    #print(data)
    save1 = pd.DataFrame(data)
    save1.to_csv('data_divide.csv', index=False, header=False)
    #data.groupby("pred").count()

    return data,pred
#把孤独森林的01拆开
def divForest01(dt,pred):
    dt1 = []
    dt0 = []
    cnt = 0
    for i in pred:
        if i == -1:
            dt0.append(dt[cnt])
        else:
            dt1.append(dt[cnt])
        cnt += 1
    save0 = pd.DataFrame(dt0)
    save0.to_csv('data_divide_0.csv', index=False, header=False)

    save1 = pd.DataFrame(dt1)
    save1.to_csv('data_divide_1.csv', index=False, header=False)
    return dt0,dt1



if __name__ == '__main__':
    # for i in range(20):
    #     print("\"x"+str(i)+"\",",end = "")
    time_start = time.time()
    dt = loadTraindata(r'C:\Users\smh\Desktop\论文\bands3.dat')
    data,pred = divDataSet(dt)
    divForest01(data,pred)
    time_end = time.time()
    print('totally cost', time_end - time_start, '秒')

