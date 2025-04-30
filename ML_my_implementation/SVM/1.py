import random
import numpy as np
import matplotlib.pyplot as plt





def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while j == i:
        j = random.randint(0, m-1)
    return j

def clipAlpha(aj, H, L):
    aj = aj.item() if isinstance(aj, np.matrix) else aj
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.matrix(dataMatIn)
    labelMat = np.matrix(classLabels).T
    b = 0
    m, n = dataMatrix.shape
    alphas = np.matrix(np.zeros((m, 1)))
    iter_num = 0

    while iter_num < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 计算fXi和Ei
            fXi = (np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)).item() + b
            Ei = fXi - labelMat[i].item()

            # 检查是否违反KKT条件
            if (labelMat[i].item() * Ei < -toler and alphas[i,0].item() < C) or \
               (labelMat[i].item() * Ei > toler and alphas[i,0].item() > 0):
                j = selectJrand(i, m)
                fXj = (np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j,:].T)).item() + b
                Ej = fXj - labelMat[j].item()
                alphaIold = alphas[i,0].copy()
                alphaJold = alphas[j,0].copy()

                # 计算L和H
                if labelMat[i].item() != labelMat[j].item():
                    L = max(0, alphaJold - alphaIold)
                    H = min(C, C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaJold + alphaIold - C)
                    H = min(C, alphaJold + alphaIold)
                if L == H:
                    print("L == H")
                    continue

                # 计算eta
                eta = 2.0 * (dataMatrix[i,:] * dataMatrix[j,:].T).item() - \
                      (dataMatrix[i,:] * dataMatrix[i,:].T).item() - \
                      (dataMatrix[j,:] * dataMatrix[j,:].T).item()
                if eta >= 0:
                    print("eta >= 0")
                    continue

                # 更新alpha_j
                alphas[j,0] -= labelMat[j].item() * (Ei - Ej) / eta
                alphas[j,0] = clipAlpha(alphas[j,0], H, L)

                if abs(alphas[j,0] - alphaJold) < 0.00001:
                    print("alpha_j变化太小")
                    continue

                # 更新alpha_i
                alphas[i,0] += labelMat[j].item() * labelMat[i].item() * (alphaJold - alphas[j,0].item())

                # 更新b
                b1 = b - Ei - labelMat[i].item() * (alphas[i,0].item() - alphaIold) * (dataMatrix[i,:] * dataMatrix[i,:].T).item() - \
                     labelMat[j].item() * (alphas[j,0].item() - alphaJold) * (dataMatrix[i,:] * dataMatrix[j,:].T).item()
                b2 = b - Ej - labelMat[i].item() * (alphas[i,0].item() - alphaIold) * (dataMatrix[i,:] * dataMatrix[j,:].T).item() - \
                     labelMat[j].item() * (alphas[j,0].item() - alphaJold) * (dataMatrix[j,:] * dataMatrix[j,:].T).item()

                if 0 < alphas[i,0].item() < C:
                    b = b1
                elif 0 < alphas[j,0].item() < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print(f"第{iter_num}次迭代 样本:{i}, alpha优化次数:{alphaPairsChanged}")

        if alphaPairsChanged == 0:
            iter_num += 1
        else:
            iter_num = 0
        print(f'迭代次数: {iter_num}')

    return b, alphas



"""
函数说明:计算w

Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
    alphas - alphas值
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-09-23
"""


def get_w(dataMat, labelMat, alphas):
    # 转换数据格式并过滤支持向量
    alphas = np.array(alphas).flatten()  # 展平为一维数组
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)

    # 仅保留alpha > 0的支持向量
    idx = alphas > 1e-5  # 过滤小量alpha避免数值误差
    support_vectors = dataMat[idx]
    support_labels = labelMat[idx]
    support_alphas = alphas[idx]

    # 计算权重向量w = Σ(alpha_i * y_i * x_i)
    w = np.dot(support_vectors.T, support_alphas * support_labels)
    print("权重向量w:", w)
    return w



"""
函数说明:分类结果可视化

Parameters:
    dataMat - 数据矩阵
    w - 直线法向量
    b - 直线解决
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-09-23
"""
def showClassifer(dataMat, w, b):
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()




if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)
