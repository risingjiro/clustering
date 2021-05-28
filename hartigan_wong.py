print("I study git")

import random
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools

def hartigan_wong(A, k, iteration=20): 

    def NC(l): 
        nc = A[IC1==l].shape[0]
        return nc

    def D(i, l): 
        distance = np.sum((centroids[l] - A[i]) ** 2)
        return distance

    # step4 OPTRA-stage
    def step4(IC1, IC2, live, centroids): 
        tmp_live = []
        # for _ in range(iteration): 
        for i in range(m): 
            L1 = IC1[i]

            R2_list = []
            L2_list = []

            if L1 in live: 
                    
                for l in range(k): 

                    # L1以外のクラスターについて計算するため、L1の場合、for文をスキップする
                    if l == L1: 
                        continue

                    R2_list.append((NC(l) * D(i, l) / (NC(l) + 1)))
                    L2_list.append(l)
            else: 
                for l in range(k): 
                    if l in live: 
                        continue
                    R2_list.append((NC(l) * D(i, l) / (NC(l) + 1)))
                    L2_list.append(l)

            # L2がなかったら
            if len(L2_list) == 0: 
                continue

            
            L2 = L2_list[R2_list.index(min(R2_list))]


            # この式を満たす場合、点Iは移動せず、L2がIC2になるだけ
            if min(R2_list) >= (NC(L1) * D(i, L1) / (NC(L1) - 1)): 
                IC2[i] = L2
                # このときは移動が起きていないため、liveに追加しない

            # このとき、点IはクラスターL2に移動する。そして、L1がIC2になる。
            else: 
                IC1[i] = L2
                IC2[i] = L1

                # 点の移動が起きたため、クラスター中心を再計算。
                for l in range(k): 
                    centroids[l] = A[IC1 == l].mean(axis=0)

                # 移動に関わった2つのクラスターはliveに属する    
                if L1 not in tmp_live: 
                    tmp_live.append(L1)
                if L2 not in tmp_live: 
                    tmp_live.append(L2)

        live = tmp_live   
        tmp_live = []

        if len(live) == 0: 
            print('break! because the live set is empty!')
            return IC1, live, centroids
            # break
        else: 
            step6(IC1, IC2, live, centroids)

        return IC1, centroids, live

    # step6 QTRAN stage
    def step6(IC1, IC2, live, centroids):
        for i in range(m): 
            L1 = IC1[i]
            L2 = IC2[i]

            tmp_IC1 = IC1

            R1 = (NC(L1) * D(i, L1)) / (NC(L1) - 1)
            R2 = (NC(L2) * D(i, L2)) / (NC(L2) + 1)

            if R1 >= R2: 
                tmp_IC1[i] = L2
                IC2[i] = L1

                # クラスター中心を更新する
                for l in range(k): 
                    centroids[l] = A[tmp_IC1 == l].mean(axis=0)

        if (IC1 == tmp_IC1).all: 
            IC1 == tmp_IC1
            step4(IC1, IC2, live, centroids)
        else: 
            IC1 = tmp_IC1
            step6(IC1, IC2, live, centroids)

    # インプットのデータより、M(データ数), N(データの次元)を抽出して変数にする
    m = A.shape[0]
    n = A.shape[1]

    # 初期クラスター中心を求める。平均値で並び替える
    # まず、平均値を求める
    ave = np.mean(A, axis = 0)
    # 平均値からの距離を置いておく配列
    d_from_ave = []
    for i in range(m): 
        d_from_ave.append(np.sum((ave - A[i]) ** 2))
    # 平均値からの距離で並び替えたindexを保存。
    idx = np.argsort(d_from_ave)
    centroids = []
    # 該当するindexのデータを初期クラスター中心とする。ん
    for l in range(k): 
        lmk = l * (m / k)
        c_factor = A[idx==lmk] # クラスター中心
        c_factor = list(itertools.chain.from_iterable(c_factor)) # なぜか、二重リストになるので、1重に変更する
        c_factor = np.array(c_factor) # ndarrayに変更
        centroids.append(c_factor)

    # step1 点iが最も近いクラスター中心をIC1, 次に近いものをIC2とする。そして、点iはクラスターIC1に割り当てられる。        
    IC1 = np.full(m, 1000)
    IC2 = np.full(m, 1000)
    # 変更後のIC1を入れておく配列
    tmp_IC1 = np.full(m, 1000)

    for i in range(m): 
        distances = np.sum((centroids - A[i]) ** 2, axis=1)
        IC1[i] = np.argsort(distances)[0]
        IC2[i] = np.argsort(distances)[1]

    # step2 クラスター中心を更新する。
    for l in range(k): 
        centroids[l] = A[IC1 == l].mean(axis=0)

    # step3 最初は、全てのクラスターがthe live setに属する
    live = []
    for l in range(k): 
        live.append(k)
    tmp_live = []


    IC1, centroids, live = step4(IC1, IC2, live, centroids)
    return IC1, centroids, live

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

def wss(data, clusters, centroids, k): 
    wss = 0

    for l in range(k): 
        data_l = data[clusters == l]
        for i in data_l: 
            wss += np.sum((centroids[l] - i) ** 2)

    return wss


if __name__ == '__main__': 
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = iris.data
    import time
    start = time.time()
    clusters, centroids, live = hartigan_wong(data, 3)
    end = time.time()
    print("実行にかかった時間は{}".format(end - start))

    # wssを表示
    w = wss(data, clusters, centroids, 3)
    print(w)

    # 結果の可視化
    iris_visu(data, clusters)