import random
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
def macqueen(data, n_clusters, max_iter=10, rs_for_initial_values=0): 
    np.random.seed(rs_for_initial_values)

    #k個の個体をランダムに選ぶ
    #k個の個体の、indexと値を取得
    #initial_values = random.sample(list(enumerate(data)), k=n_clusters)
    #initial_values = np.random.choice(list(enumerate(data)), k=n_clusters, replace=False)

    random_index = np.random.randint(0, data.shape[0], n_clusters)
    initial_values = data[random_index]
    centroids = list([x for x in initial_values])
    initial_index = list([x for x in random_index])

    #centroids = list([x[1] for x in initial_values])

    #initial_index = [x[0] for x in initial_values]
    #クラスターを保存する
    clusters = np.full(data.shape[0], 100)
    new_clusters = np.full(data.shape[0], 100)
    #その個体を１つの個体からなるクラスタとする
    for c, index in enumerate(initial_index): 
        clusters[index] = c

  #ステップ２残りの個体を各クラスターに当てはめる
    for i in range(data.shape[0]): 
        distances = np.sum((centroids - data[i]) ** 2, axis=1)
        clusters[i] = np.argsort(distances)[0]
    for j in range(n_clusters): 
        #個体をクラスターに当てはめるたびに重心を再計算する
        centroids[j] = data[clusters == j].mean(axis=0)


  #ステップ３(収束するまで繰り返す)
  #重心を再計算する
    for _ in range(max_iter): 
        for j in range(n_clusters): 
            centroids[j] = data[clusters == j].mean(axis=0)
    #そしてクラスターを割り振り直す
        for i in range(data.shape[0]): 
            distances = np.sum((centroids - data[i]) ** 2, axis=1)
            new_clusters[i] = np.argsort(distances)[0]

    #クラスターが変化しなくなったら終了
        if np.allclose(clusters, new_clusters): 
            print('break')
            break

    clusters = new_clusters

    return clusters, centroids

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
    

if __name__ == '__main__': 
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = iris.data
    import time

    # macqueen
    start = time.time()
    clusters, centroids = macqueen(data, 3)
    end = time.time()
    print("実行にかかった時間は{}".format(end - start))

    # wssを表示
    w = wss(data, clusters, centroids, 3)
    print("MacQueenのwss(クラスター内分散の和)は{}です。".format(w))

    # 結果の可視化
    iris_visu(data, clusters)