from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re

"""
	dataset: https://research.fb.com/downloads/babi/
	Dataset format:
		number Story sentences.....
		number Question        Answer       Supporting int(related lines):
"""

challenges = {
	    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
	    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
	}
challenge_type = 'two_supporting_facts_10k' # train for...
challenge = challenges[challenge_type]

def tokenization(sentence):
	# tokenize each sentence
	return [x.strip() for x in re.split('(\W+)?', sentence) if x.strip()]

def split_stories(lines):
	# data = final array with story, question, answer
	# story = set of all sentences for each
	data, story = [], []
	for line in lines:
		line = line.decode('utf-8').strip()
		nid, line = line.split(' ', 1)
		nid = int(nid)
		if nid == 1:
			# new task, new story...
			story = []
		if '\t' in line:
			q, a, supporting = line.split('\t')
			q = tokenization(q)
			
			substory = [x for x in story if x]
			data.append((substory, q, a))
			story.append('')
		else:
			sent = tokenization(line)
			story.append(sent)
	return data

def get_stories(file, max_length=None):
	"""
		Given a file name, read the file, retrieve the stories,
		and then convert the sentences into a single story.
		If max_length is supplied,
		any stories longer than max_length tokens will be discarded.
	"""
	data = split_stories(file.readlines()) # gives all sentences
	flatten = lambda data: reduce(lambda x, y: x + y, data) # look for help(reduce)
	data = [(flatten(story), question, answer) for story, question, answer in data]
	return data

def vectorization(data, word_idx, story_maxlen, query_maxlen):
	"""
		data is set of sentences of story
		word_idx: map between integer and words in story
		story_maxlen, query_maxlen: max length for constructing
		embeding matrix for LSTM/GRU cells 
	"""
	Xstorys = []
	Xquestions = []
	Yanswers = []
	for story, query, answer in data:
		xstory = [word_idx[w] for w in story]
		xqestion = [word_idx[w] for w in query]

		# let's not forget that index 0 is reserved
		yanswer = np.zeros(len(word_idx) + 1)
		yanswer[word_idx[answer]] = 1 # onehot vector for answers 
		
		# group of all stories, questions, answers
		Xstorys.append(xstory)
		Xquestions.append(xqestion)
		Yanswers.append(yanswer)

	return (pad_sequences(Xstorys, maxlen=story_maxlen),
	        	pad_sequences(Xquestions, maxlen=query_maxlen), 
	        	np.array(Yanswers))

def download_dataset(path=None):
	"""
		downloads dataset in ~/.keras/datasets/ unless specified
	"""
	try:
		path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
	except:
		print('Error downloading dataset, please download it manually:\n'
	      		'$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
	      		'$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')

	# returns tar object...
	return tarfile.open(path)

def test_train_split(challenge):
	tar = download_dataset() # tar object

	print('Extracting stories for the challenge:')
	train_stories = get_stories(tar.extractfile(challenge.format('train')))
	test_stories = get_stories(tar.extractfile(challenge.format('test')))

	# get statistics embeding matrix
	vocab = set()
	for story, q, answer in train_stories + test_stories: vocab |= set(story + q + [answer])
	vocab = sorted(vocab)

	# Reserve 0 for masking via pad_sequences
	vocab_size = len(vocab) + 1
	story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
	query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

	return (train_stories, test_stories,vocab), (story_maxlen, query_maxlen, vocab_size)

def print_statistics():
	(train_stories, test_stories, vocab), \
	(story_maxlen, query_maxlen, vocab_size) = test_train_split(challenge)

	print('-'*100)
	print('Vocab size:', vocab_size, 'unique words')
	print('Story max length:', story_maxlen, 'words')
	print('Query max length:', query_maxlen, 'words')
	print('Number of training stories:', len(train_stories))
	print('Number of test stories:', len(test_stories))
	print('-'*100)
	print('Here\'s what a "story" tuple looks like (input, query, answer):')
	print(train_stories[0])
	
	word_idx = dict((c, i + 1) for i, c in enumerate(vocab))# int to string map

	inputs_train, queries_train, answers_train = vectorization(train_stories,
	                                                           word_idx,
	                                                           story_maxlen,
	                                                           query_maxlen)
	inputs_test, queries_test, answers_test = vectorization(test_stories,
	                                                        word_idx,
	                                                        story_maxlen,
	                                                        query_maxlen)
	print('-'*100)
	print('inputs_train shape:', inputs_train.shape)
	print('inputs_test shape:', inputs_test.shape)

	print('queries_train shape:', queries_train.shape)
	print('queries_test shape:', queries_test.shape)

	print('answers onehot vector shape:')
	print('answers_train shape:', answers_train.shape)
	print('answers_test shape:', answers_test.shape)
	return (inputs_train, queries_train, answers_train), (inputs_test, queries_test, answers_test)

class Network(object):
	def __init__(self, 
		inputs_train,
		inputs_test,
		queries_train,
		queries_test,
		answers_train,
		answers_test):

		self._inputs_train = inputs_train
		self._queries_train = queries_train
		self._answers_train = answers_train
		self._inputs_test = inputs_test
		self._queries_test = queries_test
		self._answers_test = answers_test

	def model(self):
		# input placeholders
		input_sequence = Input((self._inputs_train.shape[1], ))
		question = Input((self._queries_train.shape[1], ))

		# encoders
		# stroy encoding...
		input_encoder_m = Sequential()
		input_encoder_m.add(Embedding(input_dim=self._answers_train.shape[1],
		                              output_dim=64))
		input_encoder_m.add(Dropout(0.3))

		# query encoding...
		input_encoder_c = Sequential()
		input_encoder_c.add(Embedding(input_dim=self._answers_train.shape[1],
		                              output_dim=self._queries_train.shape[1]))
		input_encoder_c.add(Dropout(0.3))
		# output: (samples, story_maxlen, query_maxlen)

		# embed the question into a sequence of vectors
		question_encoder = Sequential()
		question_encoder.add(Embedding(input_dim=self._answers_train.shape[1],
		                               output_dim=64,
		                               input_length=self._queries_train.shape[1]))
		question_encoder.add(Dropout(0.3))
		# output: (samples, query_maxlen, embedding_dim)

		input_encoded_m = input_encoder_m(input_sequence)
		input_encoded_c = input_encoder_c(input_sequence)
		question_encoded = question_encoder(question)

		match = dot([input_encoded_m, question_encoded], axes=(2, 2))
		match = Activation('softmax')(match)

		# add the match matrix with the second input vector sequence
		response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
		response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

		# concatenate the match matrix with the question vector sequence
		answer = concatenate([response, question_encoded])

		# the original paper uses a matrix multiplication for this reduction step.
		# we choose to use a RNN instead.
		lstm_rev = Bidirectional(LSTM(32, return_sequences=True, name='Ans_LSTM_reverse'))
		lstm_for = Bidirectional(LSTM(32, return_sequences=False, name='Ans_LSTM_forward'))
		answer = lstm_rev(answer)  # "reverse" pass goes first
		answer = lstm_for(answer)

		answer = Dropout(0.3)(answer)
		answer = Dense(self._answers_train.shape[1])(answer)  # (samples, self._answers_train.shape[1])
		answer = Activation('softmax')(answer)

		# build the final model
		model = Model([input_sequence, question], answer)
		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])
		
		return model

	# model saving:	
	def train(self):
		model = self.model()
		model.fit([self._inputs_train, self._queries_train], self._answers_train,
				batch_size=32,
				epochs=120,
				validation_data=([self._inputs_test, self._queries_test],self._answers_test))
		pass

	def test(self):
		model = self.model()
		model.predict()

(inputs_train, queries_train, answers_train), (inputs_test, queries_test, answers_test) = print_statistics()
network = Network(inputs_train, inputs_test, queries_train,  queries_test, answers_train, answers_test)
network.train()