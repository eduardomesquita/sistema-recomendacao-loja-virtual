#pip3 install pandas
#pip3 install -U numpy scipy scikit-learn
import sys
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



class Train(object):
	
	def __init__(self, path):
		try:
			self._ds = pd.read_csv(path)
			self._results = {}
		except:
			exit()

	def train(self):
		tf = TfidfVectorizer(analyzer='word', 
				ngram_range=(1, 3), 
				min_df=0, 
				stop_words='english')

		tfidf_matrix = tf.fit_transform(self._ds['description'])
		cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

		for idx, row in self._ds.iterrows():
		    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
		    similar_items = [(cosine_similarities[idx][i], self._ds['id'][i]) for i in similar_indices]

		    # First item is the item itself, so remove it.
		    # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
		    self._results[row['id']] = similar_items[1:]

	def getItem(self, _id):
    		return self._ds.loc[self._ds['id'] == _id]['description'].tolist()[0].split(' - ')[0]


	def recommend(self, item_id, num):
		print('Recomendando ' + str(num) + ' produtos similares com: ' + self.getItem(item_id) + '...')
		print('-' * 5)
		recs = self._results[item_id][:num]
		for rec in recs:
			print('Recomendado: ' + self.getItem(rec[1]) + ' (pontuação:' + str(rec[0]) + ')')
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--item', type=int, help='Id do item')
	parser.add_argument('--qtd', type=int, help='Quantidade de recomendações')
	parser.add_argument('--path', type=str, help='Caminho do arquivo dataSet')
	args = parser.parse_args()

	t = Train(args.path)

	t.train()
	t.recommend(args.item, args.qtd)

