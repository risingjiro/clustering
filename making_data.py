def making_simulation_data(n_clusters, dimension, mean, cov, size, percentage_of_random): 
	'''
	when columns = 2 : 
	mean = [[x1, y1], [x2, y2]]
	cov = [[[s11, s12],[s12, s22]], [[s33, s34],[s34, s44]]]
	size = [200, 200]

	when columns = 3: 
	mean = [[x1, y1, z1], [x2, y2, z2]]
	cov = [[s111, s112, s113],[s211, s212, s213],[s311, s312]], []
	'''


	import numpy as np
	np.random.seed(0)

	dat = np.zeros((1, dimension))


	for m, c, s in zip(mean, cov, size): 
		dat = np.r_[dat, (np.random.multivariate_normal(m, c, s))]
	data = np.delete(dat, 0, axis=0)

	#ランダムに欠損を与える
	ranx = np.random.randint(0, data.shape[0], int(data.shape[0] * percentage_of_random))
	rany = np.random.randint(0, data.shape[1], int(data.shape[0] * percentage_of_random))

	for x, y in zip(ranx, rany): 
		data[x, y] = np.nan

	import pandas as pd
	df_data = pd.DataFrame(data)

	#観測値と欠損値に分ける
	complete_data = df_data[~np.isnan(data).any(axis=1)]
	missing_data = df_data[np.isnan(data).any(axis=1)]

	print('観測値の数は{}, 欠損値の数は{}'.format(len(complete_data), len(missing_data)))

	return data