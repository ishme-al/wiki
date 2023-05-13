from collections import Counter, defaultdict
# defaultdict
from math import log
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from numpy.linalg import norm
# from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def read_docs(file):
		'''
		Reads the corpus into a list of Documents
		'''
		docs = []  # empty 0 index
		with open(file) as f:
				for line in f:
						line = line.strip().split('\t')
						line[2] = list(map(str.lower, line[2][4:-5].split()))
						index = None
						for i in range(len(line[2])):
								if line[2][i][:3] == '.x-':
										index = i
										break

						# print(line[2])
						# docs.append(Document(int(line[0]), int(line[1]), list(map(str.lower, word_tokenize(line[2])))))
						docs.append(Document(int(line[0]), int(line[1]), index, line[2]))

		return docs

def read_docsLR(file):
		docs = []  # empty 0 index
		with open(file) as f:
				for line in f:
						line = line.strip().split('\t')
						line[2] = list(map(str.lower, line[2][4:-5].split()))
						index = None
						for i in range(len(line[2])):
								if line[2][i][:3] == '.x-':
										index = i
										break
						if index is not None:
								if index > 0:
										line[2][index-1] = '.l-' + line[2][index-1]
								if index < len(line[2])-1:
										line[2][index+1] = '.r-' + line[2][index+1]

						docs.append(Document(int(line[0]), int(line[1]), index, line[2]))

		return docs


def stem_doc(doc):
		return list(map(stemmer.stem, doc))

def stem_docs(docs):
		return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc):
		return list(filter(lambda x: x.lower() not in stop_words, doc))

def remove_stopwords(docs):
		return [remove_stopwords_doc(doc) for doc in docs]


### Term-Document Matrix

# class TermWeights(NamedTuple):
# 		author: float
# 		title: float
# 		keyword: float
# 		abstract: float

def compute_doc_freqs(docs):
		'''
		Computes document frequency, i.e. how many documents contain a specific word
		'''
		freq = Counter()
		for doc in docs:
				freq.update(set(doc))
		return freq

def compute_tf(doc):
		vec = defaultdict(float)
		for i in range(len(doc)):
			vec[doc[i]] += 1
		return dict(vec)  # convert back to a regular dict


def compute_tfidf(doc, doc_freqs):
		tf = compute_tf(doc)
		dic = {}
		for word in tf:
				if doc_freqs[word] == 0:
						dic[word] = tf[word] * log(len(doc_freqs) / 1)
				else:
						dic[word] = tf[word] * log(len(doc_freqs) / doc_freqs[word])
		return dic

def compute_boolean(doc, doc_freqs, weights):
		tf = compute_tf(doc, doc_freqs, weights)
		return {word: 1 for word in tf}



### Vector Similarity
def dictdot(x, y):
		'''
		Computes the dot product of vectors x and y, represented as sparse dictionaries.
		'''
		keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
		return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
		'''
		Computes the cosine similarity between two sparse term vectors represented as dictionaries.
		'''
		num = dictdot(x, y)
		if num == 0:
				return 0
		return num / (norm(list(x.values())) * norm(list(y.values())))

def dice_sim(x, y):
		num = 2 * dictdot(x, y)
		if num == 0:
				return 0
		return num / (sum(x.values()) + sum(y.values()))

def jaccard_sim(x, y):
		num = dictdot(x, y)
		if num == 0:
				return 0
		den = sum(x.values()) + sum(y.values()) - num
		if den == 0:
				den = 0.001
		return num / den

def overlap_sim(x, y):
		num = dictdot(x, y)
		if num == 0:
				return 0
		return num / min(sum(x.values()), sum(y.values()))


### Precision/Recall

def interpolate(x1, y1, x2, y2, x):
		m = (y2 - y1) / (x2 - x1)
		b = y1 - m * x1
		return m * x + b

def precision_index(index, results, relevant) -> float:
		c = index
		for doc in results:
				if doc in relevant:
						c -= 1
						if c == 0:
								return index / (results.index(doc)+1)
		raise Exception('precision index error')

def mean_precision1(results, relevant):
		return (precision_at(0.25, results, relevant) +
				precision_at(0.5, results, relevant) +
				precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant):
		return sum(precision_at(recall, results, relevant) for recall in np.linspace(0.1, 1, 10)) / 10

def norm_recall(results: list, relevant):
		ret = sum(results.index(doc)+1 for doc in relevant) - len(relevant)*(len(relevant)-1)/2
		ret /= len(relevant) * (len(results) - len(relevant))
		return 1 - ret

def norm_precision(results: list, relevant):
		from math import log10
		N = len(results)
		rel = len(relevant)
		ret = sum(log10(results.index(doc)+1) for doc in relevant) - sum(log10(i) for i in range(1, rel+1))
		ret /= N * log10(N) - (N - rel) * log10(N - rel) - rel * log10(rel)
		return 1 - ret


### Extensions

### Search

# from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction import DictVectorizer

def similarity(a, b):
	dv = DictVectorizer()
	doc_vectors = process_docs([a,b])
	dv.fit([d for d in doc_vectors])
	# doc_vectors = np.concatenate([dv.transform(d).toarray() for d in doc_vectors])

	# v1 = defaultdict(float)
	# v2 = defaultdict(float)
	# for doc, vector in zip(docs, doc_vectors):
	#       for k, v in vector.items():
	#               if doc.label == 1:
	#                       v1[k] += v
	#               else:
	#                       v2[k] += v
	# for k, v in v1.items():
	#       v1[k] = v / len(doc_vectors)
	# for k, v in v2.items():
	#       v2[k] = v / len(doc_vectors)

	# for doc, vector in zip(docs, doc_vectors):
	return cosine_sim(doc_vectors[0], doc_vectors[1])

def process_docs(docs):
		docs = list(map(word_tokenize, docs))
		docs = remove_stopwords(docs)
		docs = stem_docs(docs)

		doc_freqs = compute_doc_freqs(docs)
		vecs = [compute_tfidf(doc, doc_freqs) for doc in docs]
		return vecs