import time
import random

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.manifold import TSNE

# (539, 20)
#加载数据
def loadTraindata(str):
    f = open(str, encoding='utf-8')
    sentimentlist = []
    for line in f:
        s = line.replace('?', 'NaN').strip().split(',')
        sentimentlist.append(s)
    f.close()
    dt = np.array(sentimentlist)
    return dt

#预处理训练集
def preprocessTrainData(dt):
    train_siz = int  (dt.shape[0])
    X = dt[:, 0:19]
    y = dt[:, -1]
    Y = [0 if ((i == 'noband') or (i=='0')) else 1 for i in y]

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    X = imp.transform(X)
    X_train = X[0:train_siz, :]
    Y_train = Y[0:train_siz]

    # 随机抽样
    random.seed(5)
    sample_list = random.sample(range(0, train_siz), train_siz)

    x_t = [X_train[i] for i in sample_list]
    y_t = [Y_train[i] for i in sample_list]

    return X_train,Y_train

#预测处理 测试集
def preprocessTestData():
    f = open(r'C:\Users\smh\Desktop\论文\bands_test.dat', encoding='utf-8')
    sentimentlist = []
    for line in f:
        s = line.replace('?', 'NaN').strip().split(',')
        sentimentlist.append(s)
    f.close()
    dt = np.array(sentimentlist)

    X = dt[:, 0:19]
    y = dt[:, -1]
    Y = [0 if i == 'noband' else 1 for i in y]

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    X = imp.transform(X)
    return X,Y

# 预测
def predict(x_t,y_t,X_test,Y_test):
    clf = svm.SVC(C=1, kernel='poly', max_iter=4)
    clf.fit(x_t, y_t)
    result = clf.predict(X_test)
    rate = clf.score(X_test, Y_test)
    #print(result)
    #print(Y_test)
    #print(clf.score(X_test, Y_test))
    # result = clf.predict([[40.0,65,0.2,15.0,80,0.625,50,10.4,1640,47.2,42.4,1.0,0.0,3.0,1.5,32.0,40,103.13,100]])
    # print(result)
    return rate


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    # """
    # :param data:数据集
    # :param label:样本标签
    # :param title:图像标题
    # :return:图像
    # """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    print("data shape:",data.shape)
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1((label[i]+3) / 10),
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks()  # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig

def drawPicture(data,label):
    # 画图
    ts = TSNE(n_components=2, init='pca', random_state=0)
    reslut = ts.fit_transform(data)
    # 调用函数，绘制图像
    fig = plot_embedding(reslut, label, 't-SNE Embedding of digits')
    # 显示图像
    plt.show()

def shapley(a,b,c,ab,ac,bc,abc):
    #a-b-c
    #x1=(a)+(ab-a)+(abc-ab)
    a1 = a
    b1 = ab-a
    c1 = abc-ab
    #a-c-b
    #x2=(a)+(ac-a)+(abc-ac)
    a2 = a
    b2 = ac-a
    c2 = abc-ac

    #b-c-a
    #x3=(abc-bc)+(b)+(bc-b)
    a3 = abc-bc
    b3 = b
    c3 = b

    #b-a-c
    #x4=(ab-b)+(b)+(abc-ab)
    a4 = ab-b
    b4 = b
    c4 = abc-ab

    #c-b-a
    #x5=(abc-bc)+(bc-c)+(c)
    a5 = abc-bc
    b5 = bc-c
    c5 = c

    #c-a-b
    #x6=(ac-c)+(abc-ac)+(c)
    a6 = ac-c
    b6 = abc-ac
    c6 = c
    print("a1----a6",a1,a2,a3,a4,a5,a6)
    print("b1----b6",b1,b2,b3,b4,b5,b6)
    print("c1----c6",c1,c2,c3,c4,c5,c6)

    A = (a1+a2+a3+a4+a5+a6)/6
    B = (b1+b2+b3+b4+b5+b6)/6
    C = (c1+c2+c3+c4+c5+c6)/6

    return A,B,C

#获得训练的准确率
def getTrainRate(str):
    dt = loadTraindata(str)
    x_t, y_t = preprocessTrainData(dt)
    X_test, Y_test = preprocessTestData()

    result = predict(x_t, y_t, X_test, Y_test)
    #drawPicture(x_t,y_t)
    return result

#对比实验1
def radomDate1():
    a = getTrainRate(r'C:\Users\smh\Desktop\论文\bands5.dat')
    b = getTrainRate(r'C:\Users\smh\Desktop\论文\bands6.dat')
    c = getTrainRate(r'C:\Users\smh\Desktop\论文\bands4.dat')
    ab = getTrainRate(r'C:\Users\smh\Desktop\论文\bands3.dat')
    ac = getTrainRate(r'C:\Users\smh\Desktop\论文\bands7.dat')
    bc = getTrainRate(r'C:\Users\smh\Desktop\论文\bands8.dat')
    abc = getTrainRate(r'C:\Users\smh\Desktop\论文\bands2.dat')
    print("对比实验1：----------------------------")
    print("rate", a, b, c, ab, ac, bc, abc)
    a, b, c = shapley(a, b, c, ab, ac, bc, abc)
    print('shapley value =', a, b, c)

#对比实验2
def radomDate2():
    a = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsA.dat')
    b = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsB.dat')
    c = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsC.dat')
    ab = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsAB.dat')
    ac = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsAC.dat')
    bc = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsBC.dat')
    abc = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsABC.dat')
    print("对比实验2：----------------------------")
    print("rate", a, b, c, ab, ac, bc, abc)
    a, b, c = shapley(a, b, c, ab, ac, bc, abc)
    print('shapley value =', a, b, c)


if __name__ == '__main__':
    time_start = time.time()
    #a = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsABC.dat')
    #b = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsB.dat')
    #c = getTrainRate(r'C:\Users\smh\Desktop\论文datanew\bandsC.dat')
    #radomDate1()
    #radomDate2()
    a, b, c =shapley(0.71, 0.53, 0.42, 0.82, 0.78, 0.64, 0.92)
    print('shapley value =', a, b, c)
    time_end = time.time()
    print('totally cost', time_end - time_start, '秒')
