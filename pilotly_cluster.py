import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.decomposition import LatentDirichletAllocation

'''
HOW TO USE:
in terminal,
python pilotly_cluster.py [filepath] [number_of_clusters]

will output two files:
'cluster_output.csv': each message with a cluster assigned
'cluster_summary.csv': summary of each cluster with top words and size of each cluster

Note: file must contain "Response" column
'''

class PilotlyCluster:
	def __init__(self, unicode_error = 'UNICODE_DECODING_ERROR'):
		self.unicode_error = unicode_error
		self.messages = None
		self.messages_eng = None
		self.messages_stemmed = None
		self.tf = None
		self.feat_mtx = None
		self.dense_mtx = None
		self.small_mtx = None
		self.af = None
		self.cluster_dic = None

	def fit(self, df, num_clusters = 7, clustering_model = 'km', content_col = 'Response', ngram_range = (1,1), max_df = 1.0, min_df = 1):
		'''
		INPUT: dataframe with messages in the content col
		OUTPUT: 
		'''
		if clustering_model  == 'km':
			clustering_model = KMeans(n_clusters = num_clusters)
		elif clustering_model == 'af':
			clustering_model = AffinityPropagation(affinity = 'euclidean')
		self.clustering_model = clustering_model
		self.ngram_range = ngram_range
		self.max_df = max_df
		self.min_df = min_df
		self.messages = df[content_col].values
		print 'building tokens...'
		self.messages_stemmed = self.feat_eng()
		print 'vectorizing data...'
		self.vectorize()
		print 'clustering data...'
		self.cluster(self.clustering_model)
		self.get_topn_words()
		self.get_cluster_df()
		self.get_cluster_stats_df()


	def feat_eng(self):
		'''
		Prepares messages for vectorizing by removing encoding errors and stemming
		'''
		snowball = SnowballStemmer('english')
		messages_n = []
		for idx, message in enumerate(self.messages):
		    try:
		        messages_n.append(message.decode('utf-8').encode('ascii','ignore'))
		    except:
		        messages_n.append(self.unicode_error)

		docs_snowball = [[snowball.stem(word) for word in words.split()] for words in messages_n]
		self.messages_eng = messages_n
		self.messages_stemmed = [' '.join(x) for x in docs_snowball]
		return self.messages_stemmed

	def vectorize(self):
		'''
		Creates feature matrix using TFIDF from messages
		'''
		stopw = set(stopwords.words('english'))
		self.tf = TfidfVectorizer(stop_words = stopw, ngram_range = self.ngram_range, 
			min_df = self.min_df, max_df = self.max_df)
		self.feat_mtx = self.tf.fit_transform(self.messages_stemmed)

	def cluster(self, model):
		'''
		Builds clustering model and fits feature matrix
		'''
		self.af = model
		self.small_mtx = self.feat_mtx.toarray()
		self.af.fit(self.small_mtx)

	def get_cluster(self, cluster_num, by_size = False):
		'''
		Returns information about selected cluster and the messages that make up that cluster
		'''
		if by_size == True:
			cluster_num = self.cluster_stats_df.index.values[cluster_num]
		print 'cluster: ', cluster_num
		print 'top words: ', ', '.join(self.cluster_dic[cluster_num])
		return self.cluster_df[self.cluster_df['cluster'] == cluster_num]['messages'].values

	def get_cluster_stats_df(self):
		'''
		Returns df with top words and count of messages for each cluster
		'''
		self.cluster_stats_df = self.cluster_df.groupby('cluster').count().rename(columns = {'messages':'count'})
		self.cluster_stats_df['top words'] = self.cluster_stats_df.reset_index()['cluster'].apply(lambda x: ', '.join(self.cluster_dic[x]))
		self.cluster_stats_df = self.cluster_stats_df.sort('count', ascending = False)

	def get_cluster_df(self):
		'''
		Returns dataframe of messages with cluster
		'''
		self.cluster_df = pd.DataFrame(np.hstack((np.array(self.messages).reshape(-1,1), self.af.labels_.reshape(-1,1))))
		self.cluster_df.columns = ['messages', 'cluster']
		self.cluster_df = self.cluster_df.sort('cluster', ascending = True)
		return self.cluster_df

	def get_topn_words(self):
		'''
		Generates the top tokens appearing in each cluster
		'''
		d = {}
		labels = self.af.labels_
		vocabulary = self.tf.get_feature_names()
		feat_mtx = self.small_mtx
		for i in list(set(labels)):
			vec = feat_mtx[labels == i].sum(axis = 0)
			idx = np.argsort(vec)[:-10:-1]
			d[i] = np.array(vocabulary)[idx]
		for i in d:
			print i, ': ',
			for j in d[i]:
				print j, ', ',
			print ' '
		self.cluster_dic = d

	def get_dist_from_center(self):
		pass

class LDACluster(PilotlyCluster):
	def __init__(self, unicode_error = 'UNICODE_DECODING_ERROR'):
		PilotlyCluster.__init__(self, unicode_error)

	def cluster(self, model):
		'''
		Builds clustering model and fits feature matrix
		'''
		self.af = LatentDirichletAllocation()
		self.small_mtx = self.feat_mtx.toarray()
		self.topic_mtx = self.af.fit_transform(self.small_mtx)
		self.af.labels_ = np.array([np.argsort(x)[-1] for x in self.topic_mtx])
		self.label_weights = np.array([x[np.argsort(x)][-1] for x in self.topic_mtx])

	def get_cluster_df(self):
		'''
		Returns dataframe of messages with cluster
		'''
		self.cluster_df = pd.DataFrame(np.hstack((np.array(self.messages).reshape(-1,1), self.af.labels_.reshape(-1,1), 
			self.label_weights.reshape(-1,1))))
		self.cluster_df.columns = ['messages', 'cluster', 'weight']
		self.cluster_df = self.cluster_df.sort(['cluster','weight'], ascending = False)
		return self.cluster_df

	def get_cluster(self, cluster_num, by_size = False):
		'''
		Returns information about selected cluster and the messages that make up that cluster
		'''
		if by_size == True:
			cluster_num = self.cluster_stats_df.index.values[cluster_num]
		print 'cluster: ', cluster_num
		print 'top words: ', ', '.join(self.cluster_dic[cluster_num])
		for i in self.cluster_df[self.cluster_df['cluster'] == cluster_num][['messages','weight']].values:
			print i[1], ': ', i[0]



def lda_get_cluster_df():
	pass

def prep_df_topicmodel(df, min_length = 0, content_col = 'Response'):
	'''
	INPUT: df from survey with 'Response' 
	OUTPUT: df without na and with minimum 
	'''
	df = df.dropna(0, subset = [content_col])
	df = df[df[content_col].apply(lambda x: len(x.split())) > min_length]
	return df


if __name__ == '__main__':
	filename = sys.argv[1]
	if len(sys.argv) > 2:
		clusters = int(sys.argv[2])
	else:
		clusters = 7
	df = pd.read_csv(filename)
	df = prep_df_topicmodel(df, min_length = 5)
	pc = PilotlyCluster()
	pc.fit(df, num_clusters = clusters)
	pc.cluster_df.to_csv('cluster_output.csv', index = False)
	pc.cluster_stats_df.to_csv('cluster_summary.csv')