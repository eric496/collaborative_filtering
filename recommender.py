import scipy.spatial.distance as dist
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import sys
#table format
#ratings: user_id, movie_id, rating, timestamp
#toBeRated: user_id, movie_id

if __name__ == '__main__':
	#read parameters from cmd
	training_path = sys.argv[1]
	print training_path
	testing_path = sys.argv[2]
	print testing_path
	method = sys.argv[3]
	print method
	
	#read csv files and form data frames 
	ratings_table = pd.read_csv('ratings.csv', sep=',', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
	to_be_rated_table = pd.read_csv('toBeRated.csv', sep=',', header=None, names=['user_id', 'movie_id', 'rating'])

	#valid methods: 'cosine', 'jaccard', 'correlation'
	def item_based_cross_validate(ratings_table):
		rmse_list = []
		k = 10
		for i in xrange(k):
			#10-fold cross validation
			#9 folds as training and 1 fold as testing
			training = ratings_table.ix[ratings_table.index % k != i]
			testing = ratings_table.ix[ratings_table.index % k == i]
			
			#calculate the mean of all movies
			movie_id_group = training.groupby('movie_id')
			movie_mean = movie_id_group['rating'].mean()

			#create an empty user-item ndarray
			user_item = np.zeros((len(set(ratings_table['user_id'])), len(set(ratings_table['movie_id']))))
			#get all user ids
			index = list(set(ratings_table['user_id']))
			#get all movie ids
			columns = list(set(ratings_table['movie_id']))
			#create a user-item dataframe
			user_item = pd.DataFrame(user_item, index=index, columns=columns)

			#fill in ratings
			for j, row in training.iterrows():
				user_item.ix[row['user_id']][row['movie_id']] = row['rating']
			#item-item similarity matrix
			item_square = dist.squareform(dist.pdist(np.array(user_item.T), method))
			#item-item similarity dataframe
			item_item = pd.DataFrame(item_square, index=columns, columns=columns)

			est_ratings = []
			for j, row in testing.iterrows():
				print "user id: " + str(row['user_id']) + " movie id: " + str(row['movie_id'])
				#get all similar items including itself
				sim_items = pd.DataFrame(item_item[row['movie_id']], columns=[row['movie_id']])
				#get similarity between current user and all similar users
				similarity = 1 - sim_items
				#remove self
				similarity = similarity[similarity.index!=row['movie_id']]
				#get mean ratings as the predicted rating 
				est = np.around(np.mean(movie_mean[similarity.index]))
				#add the mean rating to the list
				est_ratings.append(est)
				#free memory 
				if j == testing.index[-1]:
					del sim_items, similarity, ix, est
					gc.collect()

			se = np.square(np.array(est_ratings) - np.array(testing['rating']))
			mse = np.nansum(se)/np.count_nonzero(~np.isnan(se))
			rmse = np.sqrt(mse)
			rmse_list.append(rmse)
			print "cross validation " + str(i) + " completed."
			#free memory
			del training, testing, user_item, index, columns, item_square, item_item
			gc.collect()
		print "RMSE of 10-fold cross validation: "
		print rmse_list
		
	def user_based_cross_validate(ratings_table):
		rmse_list = []
		k = 10
		for i in xrange(k):
			#10-fold cross validation
			#9 folds as training and 1 fold as testing
			training = ratings_table.ix[ratings_table.index % k != i]
			testing = ratings_table.ix[ratings_table.index % k == i]

			#create an empty user-item ndarray
			user_item = np.empty((len(set(ratings_table['user_id'])), len(set(ratings_table['movie_id']))))
			user_item[:] = np.nan
			#get all user ids
			index = list(set(ratings_table['user_id']))
			#get all movie ids
			columns = list(set(ratings_table['movie_id']))
			#create a user-item dataframe
			user_item = pd.DataFrame(user_item, index=index, columns=columns)
			#fill in ratings
			for j, row in training.iterrows():
				user_item.ix[row['user_id']][row['movie_id']] = row['rating']
			#normalize by subtracting the mean rating of each user
			user_item_T = user_item.T - np.nanmean(user_item, axis=1)
			user_item = user_item_T.T
			user_item.fillna(0, inplace=True)
			#user-user similarity matrix
			user_square = dist.squareform(dist.pdist(np.array(user_item), method))
			#user-user similarity dataframe
			user_user = pd.DataFrame(user_square, index=index, columns=index)

			est_ratings = []
			for j, row in testing.iterrows():
				print "user id: " + str(row['user_id']) + " movie id: " + str(row['movie_id'])
				#get all similar users including itself
				sim_users = pd.DataFrame(user_user[row['user_id']], columns=[row['user_id']])
				#get similar users who have rated the movie to be rated
				sim_users_rated = ratings_table[ratings_table['movie_id']==row['movie_id']]['user_id']
				#get similarity between current user and all similar users
				similarity = 1 - sim_users.ix[sim_users_rated]
				#remove self
				similarity = similarity[similarity.index!=row['user_id']]
				#calculate mean rating of similar users
				est = np.around(np.mean(user_item.ix[similarity.index][row['movie_id']]))
				#add the mean rating to the list
				est_ratings.append(est)
				#free memory
				if j == testing.index[-1]:
					del sim_users, sim_users_rated, similarity, est
					gc.collect()
			se = np.square(np.array(est_ratings) - np.array(testing['rating']))
			mse = np.nansum(se)/np.count_nonzero(~np.isnan(se))
			rmse = np.sqrt(mse)
			rmse_list.append(rmse)
			print "cross validation " + str(i) + " completed."
			#free memory
			del training, testing, user_item, index, columns, user_square, user_user
			gc.collect()
			print "RMSE of 10-fold cross validation: "
			print rmse_list

	def improved_cross_validate(ratings_table):
		rmse_list = []
		k = 10
		for i in xrange(k):
			#10-fold cross validation
			#9 folds as training and 1 fold as testing
			training = ratings_table.ix[ratings_table.index % k != i]
			testing = ratings_table.ix[ratings_table.index % k == i]

			#create an empty user-item ndarray
			user_item = np.zeros((len(set(ratings_table['user_id'])), len(set(ratings_table['movie_id']))))
			#get all user ids
			index = list(set(ratings_table['user_id']))
			#get all movie ids
			columns = list(set(ratings_table['movie_id']))
			#create a user-item dataframe
			user_item = pd.DataFrame(user_item, index=index, columns=columns)
			#fill in ratings
			for j, row in training.iterrows():
				user_item.ix[row['user_id']][row['movie_id']] = row['rating']
			#user-user similarity matrix
			user_square = dist.squareform(dist.pdist(np.array(user_item), method))
			#user-user similarity dataframe
			user_user = pd.DataFrame(user_square, index=index, columns=index)

			est_ratings = []
			for j, row in testing.iterrows():
				print "user id: " + str(row['user_id']) + " movie id: " + str(row['movie_id'])
				#get all similar users including itself
				sim_users = pd.DataFrame(user_user[row['user_id']], columns=[row['user_id']])
				#get similar users who have rated the movie to be rated
				sim_users_rated = ratings_table[ratings_table['movie_id']==row['movie_id']]['user_id']
				#get similarity between current user and all similar users
				similarity = 1 - sim_users.ix[sim_users_rated]
				#remove self
				similarity = similarity[similarity.index!=row['user_id']]
				#fill nan with zero
				#similarity.fillna(0, inplace=True)
				#get indices of 5 most similar users
				if len(sim_users_rated) > 5:
					#get 5 maximum similarity indices (user ids)
					ix = similarity.index[np.argpartition(np.array(list(similarity[row['user_id']])), -5)[-5:]]
				else:
					ix = similarity.index[np.argpartition(np.array(list(similarity[row['user_id']])), -(len(sim_users_rated) - 1))[-(len(sim_users_rated) - 1):]]
				#get mean ratings as the predicted rating 
				sim_ratings = user_item.ix[ix][row['movie_id']]
				#weighted ratings
				weighted_sim = similarity.ix[ix]/np.sum(similarity.ix[ix][row['user_id']])
				est = np.around(np.inner(sim_ratings, weighted_sim[row['user_id']]))
				#add to list
				est_ratings.append(est)
				if j == testing.index[-1]:
					del sim_users, sim_users_rated, similarity, sim_ratings, weighted_sim, est
					gc.collect()
			se = np.square(np.array(est_ratings) - np.array(testing['rating']))
			mse = np.nansum(se)/np.count_nonzero(~np.isnan(se))
			rmse = np.sqrt(mse)
			rmse_list.append(rmse)
			print "cross validation " + str(i) + " completed."
			del training, testing, user_item, index, columns, user_square, user_user
			gc.collect()
			print 'RMSE of 10-fold cross validation: '
			print rmse_list
			
	def item_based_predict_ratings(training, testing):	
		#calculate the mean of all movies
		movie_id_group = training.groupby('movie_id')
		movie_mean = movie_id_group['rating'].mean()

		#create an empty user-item ndarray
		user_item = np.zeros((len(set(training['user_id'])), len(set(training['movie_id']))))
		#get all user ids
		index = list(set(training['user_id']))
		#get all movie ids
		columns = list(set(training['movie_id']))
		#create a user-item dataframe
		user_item = pd.DataFrame(user_item, index=index, columns=columns)

		#fill in ratings
		for j, row in training.iterrows():
			user_item.ix[row['user_id']][row['movie_id']] = row['rating']
		#item-item similarity matrix
		item_square = dist.squareform(dist.pdist(np.array(user_item.T), method))
		#item-item similarity dataframe
		item_item = pd.DataFrame(item_square, index=columns, columns=columns)

		est_ratings = []
		for j, row in testing.iterrows():
			try:
				print "user id: " + str(row['user_id']) + " movie id: " + str(row['movie_id'])
				#get all similar items including itself
				sim_items = pd.DataFrame(item_item[row['movie_id']], columns=[row['movie_id']])
				#get similarity between current user and all similar users
				similarity = 1 - sim_items
				#remove self
				similarity = similarity[similarity.index!=row['movie_id']]
				#get mean ratings as the predicted rating 
				est = np.around(np.mean(movie_mean[similarity.index]))
				#print estimated rating
				row['rating'] = est
				#free memory 
				if j == testing.index[-1]:
					del sim_items, similarity, est
					gc.collect()
			except KeyError:
				pass
		testing.to_csv('result2.csv', header=False, index=False)

	def user_based_predict_ratings(training, testing):
		#create an empty user-item ndarray
		user_item = np.zeros((len(set(training['user_id'])), len(set(training['movie_id']))))
		#get all user ids
		index = list(set(ratings_table['user_id']))
		#get all movie ids
		columns = list(set(ratings_table['movie_id']))
		#create a user-item dataframe
		user_item = pd.DataFrame(user_item, index=index, columns=columns)
		#fill in ratings
		for j, row in training.iterrows():
			user_item.ix[row['user_id']][row['movie_id']] = row['rating']
		#normalize by subtracting the mean rating of each user
		#user_item_T = user_item.T - np.nanmean(user_item, axis=1)
		#user_item = user_item_T.T
		#user_item.fillna(0, inplace=True)
		#user-user similarity matrix
		user_square = dist.squareform(dist.pdist(np.array(user_item), method))
		#user-user similarity dataframe
		user_user = pd.DataFrame(user_square, index=index, columns=index)
		for j, row in testing.iterrows():
			try:
				print "user id: " + str(row['user_id']) + " movie id: " + str(row['movie_id'])
				#get all similar users including itself
				sim_users = pd.DataFrame(user_user[row['user_id']], columns=[row['user_id']])
				#get similar users who have rated the movie to be rated
				sim_users_rated = ratings_table[ratings_table['movie_id']==row['movie_id']]['user_id']
				#get similarity between current user and all similar users
				similarity = 1 - sim_users.ix[sim_users_rated]
				#remove self
				similarity = similarity[similarity.index!=row['user_id']]
				#calculate mean rating of similar users
				est = np.around(np.mean(user_item.ix[similarity.index][row['movie_id']]))
				#print estimated rating
				testing.loc[j, 'rating'] = est
				#free memory
				if j == testing.index[-1]:
					del sim_users, sim_users_rated, similarity, est
					gc.collect()
			except KeyError:
				pass
		testing.to_csv('result1.csv', header=False, index=False)

	def improved_predict_ratings(training, testing):
		#create an empty user-item ndarray
		user_item = np.zeros((len(set(training['user_id'])), len(set(training['movie_id']))))
		#get all user ids
		index = list(set(training['user_id']))
		#get all movie ids
		columns = list(set(training['movie_id']))
		#create a user-item dataframe
		user_item = pd.DataFrame(user_item, index=index, columns=columns)
		#fill in ratings
		for j, row in training.iterrows():
			user_item.ix[row['user_id']][row['movie_id']] = row['rating']
		#user-user similarity matrix
		user_square = dist.squareform(dist.pdist(np.array(user_item), method))
		#user-user similarity dataframe
		user_user = pd.DataFrame(user_square, index=index, columns=index)

		for j, row in testing.iterrows():
			try:
				print "user id: " + str(row['user_id']) + " movie id: " + str(row['movie_id'])
				#get all similar users including itself
				sim_users = pd.DataFrame(user_user[row['user_id']], columns=[row['user_id']])
				#get similar users who have rated the movie to be rated
				sim_users_rated = ratings_table[ratings_table['movie_id']==row['movie_id']]['user_id']
				#get similarity between current user and all similar users
				similarity = 1 - sim_users.ix[sim_users_rated]
				#remove self
				similarity = similarity[similarity.index!=row['user_id']]
				#knn
				#get indices of 5 most similar users
				if len(sim_users_rated) > 5:
					#get 5 maximum similarity indices (user ids)
					ix = similarity.index[np.argpartition(np.array(list(similarity[row['user_id']])), -5)[-5:]]
				else:
					ix = similarity.index[np.argpartition(np.array(list(similarity[row['user_id']])), -(len(sim_users_rated) - 1))[-(len(sim_users_rated) - 1):]]
				#get similar user ratings
				sim_ratings = user_item.ix[ix][row['movie_id']]
				#assign weights to neighbors
				weighted_sim = similarity.ix[ix]/np.sum(similarity.ix[ix][row['user_id']])
				#calculate weighted mean as predicted rating
				est = np.around(np.inner(sim_ratings, weighted_sim[row['user_id']]))
				#update rating
				testing.loc[j, 'rating'] = est
				if j == testing.index[-1]:
					del sim_users, sim_users_rated, similarity, sim_ratings, weighted_sim, est
					gc.collect()
			except KeyError:
				pass
		testing.to_csv('result3.csv', header=False, index=False)
		
	def draw_plot():
		uu_cos = [3.7562,3.7547,3.7521,3.7537,3.7507,3.7536,3.7560,3.7565,3.7530,3.7545]
		uu_corr = [3.7655,3.7623,3.7579,3.7608,3.7621,3.7586,3.7636,3.7595,3.7567,3.7621]
		uu_jac = [3.7636,3.7605,3.7575,3.7574,3.7587,3.7538,3.7626,3.7622,3.7545,3.7601]
		ii_cos = [1.0530,1.0507,1.0490,1.0472,1.0495,1.0487,1.0492,1.0520,1.0494,1.0475]
		ii_corr = [1.0478,1.0447,1.0439,1.0448,1.0483,1.0402,1.0479,1.0495,1.0447,1.0473]
		ii_jac = [1.0337,1.0333,1.0304,1.0319,1.0350,1.0307,1.0367,1.0349,1.0336,1.0331]

		user_based_cos = np.mean(uu_cos)
		user_based_corr = np.mean(uu_corr)
		user_based_jac = np.mean(uu_jac)
		item_based_cos = np.mean(ii_cos)
		item_based_corr = np.mean(ii_corr)
		item_based_jac = np.mean(ii_jac)
		im_cos = 1.0212

		N = 7
		x = np.arange(1, N+1)
		y = [user_based_cos, user_based_corr, user_based_jac, item_based_cos, item_based_corr, item_based_jac, im_cos]
		labels = ['user cos', 'user corr', 'user jacc', 'item cos', 'item corr', 'item jacc', 'my method']
		width = 0.5
		bar1 = plt.bar(x, y, width, color="blue")
		plt.ylabel('RMSE')
		plt.xticks(x + width/2.0, labels, rotation=20)
		plt.show()