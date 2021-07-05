import random
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools

from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data
import time

def wss(data, clusters, centroids, k): 
    wss = 0

    for l in range(k): 
        data_l = data[clusters == l]
        for i in data_l: 
            wss += np.sum((centroids[l] - i) ** 2)

    return wss

def iris_visu(data, clusters): 
    X = [0, 0, 0, 1, 1, 2]
    Y = [1, 2, 3, 2, 3, 3]

    plt.figure(figsize=(12, 8))
    for i, x, y in zip(range(6), X, Y): 
        plt.subplot(2, 3, i+1)
        for cluster, c in zip(range(3), 'rgb'): 
            plt.scatter(
            data[clusters==cluster][:, x], 
            data[clusters==cluster][:, y], 
            c=c
            )
            plt.xlabel(iris.feature_names[x])
            plt.ylabel(iris.feature_names[y])
            plt.autoscale()
            plt.grid()
    plt.show()

# Hartiganのアルゴリズム
def hartigan_missing(data, n_clusters, rs_for_initial_values=0):
    np.random.seed(rs_for_initial_values)
    
    #ランダムにクラスタを割り当てる
    clusters = np.random.randint(0, n_clusters, data.shape[0])

    #nc(クラスタcに属する値の数)
    def nc(cluster) : 
        #nc = len(data[clusters==cluster])
        nc = data[clusters==cluster].shape[0]
        return nc

    #ncj(クラスタcに属する, j列の欠損していない値の数)
    def ncj(cluster, j) : 
        data_cj = data[clusters==cluster][:, j]
        complete_data_cj = ~np.isnan(data_cj)
        ncj = np.count_nonzero(complete_data_cj)
        return ncj

    #x_bar cj 
    def xbar_cj(cluster, j) : 
        sum_xij = 0
        for x in data[clusters==cluster][:, j] : 
            if math.isnan(x) : #xが欠損値nanの場合にTrueを返す
                #sum_xij += 0
                pass
            else : 
                sum_xij += x
        return sum_xij / ncj(cluster, j)
    
    #判別式の右辺のシグマ以降を計算する関数
    def right_sigma(q, i, frm = 0, to = data.shape[1]):
        result = 0;
        for j in range(frm, to):
            if data[i, j] == np.nan : 
                #result += 0
                pass
            else : 
                result += (xbar_cj(q, j) - data[i, j]) ** 2
        return result

    #判別式の左辺のシグマ以降を計算する関数
    def left_sigma(p, i, frm = 0, to = data.shape[1]):
        result = 0
        for j in range(frm, to):
            if data[i, j] == np.nan : 
                #result += 0
                pass
            else : 
                result += (ncj(p, j) ** 2 / ((ncj(p, j) - 1) ** 2)) * ((xbar_cj(p, j) - data[i, j]) ** 2)
        return result
    
    #アルゴリズムの実行
    #個体が一巡する間に入れ替えが起こらなければ終了なので、入れ替えが起こらないときに、カウントする
    cnt = 0
    while cnt < data.shape[0] : 
        #判別式の実行
        for i, x in enumerate(data) : 
            #リストを初期化
            right_list = []
            q_list = []
            #pをxの属するクラスタにする
            p = clusters[i]
            for q in range(n_clusters) : 
                #pの属さないクラスタに対し、計算を行う
                if p != q : 
                    left = ((nc(p) - 1) / nc(p)) * left_sigma(p, i)
                    right = (nc(q) / (nc(q) + 1)) * right_sigma(q, i)
                    if left > right : 
                        #判別式を満たすものをリストに追加する
                        right_list.append(right)
                        q_list.append(q)
            if len(right_list) != 0 : 
                #リストが空でなければ、リストの最小値のクラスタを割り当てる
                clusters[i] = q_list[right_list.index(min(right_list))]
                #クラスタが変更されたため、カウントを０にする
                cnt = 0
            else : 
                #リストが空ならば、クラスタは変更されない。したがって、カウントする
                cnt += 1
    centroids = []
    for j in range(n_clusters): 
        centroids.append(data[clusters == j].mean(axis=0))
    return clusters, centroids

if __name__ == '__main__': 
    # Hartigan
    start = time.time()
    clusters, centroids = hartigan_missing(data, 3)
    end = time.time()
    print("実行にかかった時間は{}".format(end - start))

    # wssを表示
    w = wss(data, clusters, centroids, 3)
    print("Hartiganのwss(クラスター内分散の和)は{}です。".format(w))

    # 結果の可視化
    iris_visu(data, clusters)