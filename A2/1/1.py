import utils
import sys
import string
import random
import numpy as np
import math
import sklearn
import time
from nltk import bigrams

train_file = sys.argv[1]
test_file = sys.argv[2]
part_num = sys.argv[3]

#---------------------- Part (a) --------------------------

dict_bigram = dict()
class_word_count_bi = np.zeros(5)
class_docu_count_bi = np.zeros(5)

dict = dict()
class_word_count = np.zeros(5)
class_docu_count = np.zeros(5)

m_test = 0
correct_pred = 0

y_pred = []
y_actual = []

# Training:
def train(stem_flag=False, bigram_flag=False):
	itr = utils.json_reader(train_file)
	m = 0
	for item in itr:
		m+=1
		
		# ---- debug ----
		if m == 1000:
			break;
		if (m % 10000 == 0):
			print("Done:", m/10000.0)
		# ---------------

		stars = int(item["stars"]) - 1
		print(stars)
		if stem_flag:
			word_list = utils.getStemmedDocuments(item["text"])
		else:
			word_list = item["text"].lower().translate(str.maketrans('', '', string.punctuation)).split()

		class_word_count[stars] += len(word_list)
		class_docu_count[stars] += 1

		for word in word_list:
			if word in dict:
				dict[word][stars] += 1
			else:
				dict[word] = np.zeros(5)
				dict[word][stars] = 1

		# bigrams?
		if bigram_flag:
			word_list = list(bigrams(word_list))
			# print(word_list)
			class_word_count_bi[stars] += len(word_list)

			for word in word_list:
				if word in dict_bigram:
					dict_bigram[word][stars] += 1
				else:
					dict_bigram[word] = np.zeros(5)
					dict_bigram[word][stars] = 1

# Testing:
def test(stem_flag=False, bigram_flag=False):
	itr = utils.json_reader(test_file)
	global m_test
	global correct_pred
	for item in itr:
		m_test += 1

		stars = int(item["stars"]) - 1
		
		if stem_flag:
			word_list = utils.getStemmedDocuments(item["text"])
		else:
			word_list = item["text"].lower().translate(str.maketrans('', '', string.punctuation)).split()
		
		# ---- debug ----
		if m_test == 100:
		# 	print(item["text"])
		# 	print(word_list)
			break;
		# ---------------

		pred_stars = 0
		max_prob = -(1e9)

		for i in range(5):
		    # P(y)
			prob = math.log(class_docu_count[i])
			
			for word in word_list:
				count = dict[word][i] if word in dict else 0
	            # P(xj/y)
				prob += math.log((count + 1) / (class_word_count[i] + len(dict)))

			# bigrams?
			if bigram_flag:
				word_list = list(bigrams(word_list))
				for word in word_list:
					count = dict_bigram[word][i] if word in dict_bigram else 0
		            # P(xj/y)
					prob += math.log((count + 1) / (class_word_count_bi[i] + len(dict_bigram)))

	        # print(max_prob, prob, pred_stars)
			if(max_prob < prob):
				max_prob = prob
				pred_stars = i

	    # print(pred_stars, stars)
		y_pred.append(pred_stars)
		y_actual.append(stars)
		if(pred_stars == stars):
			correct_pred+=1

def part_a():
	train()
	test()
	accuracy = correct_pred / m_test
	print("accuracy :",accuracy)

#---------------------- Part (b) --------------------------

def part_b():
	train_itr = utils.json_reader(train_file)
	test_itr = utils.json_reader(test_file)
	
	counts = np.zeros(5)

	for item in train_itr:
		counts[int(item["stars"]) - 1] += 1
	maj_pred = np.argmax(counts) + 1

	m = 0
	rand_correct_pred = 0
	maj_correct_pred = 0

	for item in test_itr:
		m += 1
		rand = random.randint(1,5)
		stars = int(item["stars"])
		if(rand == stars):
			rand_correct_pred += 1
		if(maj_pred == stars):
			maj_correct_pred += 1

	rand_accu = rand_correct_pred / m
	print("random prediction accuracy :", rand_accu)
	maj_accu = maj_correct_pred / m
	print("majority prediction accuracy :", maj_accu)

#---------------------- Part (c) --------------------------

def part_c():
	train()
	test()
	accuracy = correct_pred / m_test
	print("accuracy :",accuracy)
	confusion_mat = sklearn.metrics.confusion_matrix(y_actual, y_pred)
	print(confusion_mat)

#---------------------- Part (d) --------------------------

def part_d():
	train(stem_flag = True)
	test(stem_flag = True)
	accuracy = correct_pred / m_test
	print("accuracy:",accuracy)
	confusion_mat = sklearn.metrics.confusion_matrix(y_actual, y_pred)
	print("confusion matrix:\n", confusion_mat)
	f1_score = sklearn.metrics.f1_score(y_actual, y_pred, average = None)
	print("F1_score:",f1_score)
	print("macro_f1_score:", sum(f1_score) / len(f1_score))

#---------------------- Part (e) --------------------------

def part_e():
	# bi-grams
	train(stem_flag = True, bigram_flag = True)
	print("\n\n\n\n\n\n")
	test(stem_flag = True, bigram_flag = True)
	accuracy = correct_pred / m_test
	print("accuracy:",accuracy)
	print(y_actual)
	print(y_pred)
	confusion_mat = sklearn.metrics.confusion_matrix(y_actual, y_pred)
	print("confusion matrix:\n", confusion_mat)
	f1_score = sklearn.metrics.f1_score(y_actual, y_pred, average = None)
	print("F1_score:",f1_score)
	print("macro_f1_score:", sum(f1_score) / len(f1_score))

	# doc = "Manish, sachin Pranav go JLKJL lkjalksdfj kjlkjasdlf!"
	# res = utils.getStemmedDocuments(doc)
	# print(res)

#---------------------- Part (g) --------------------------


# setting print option to a fixed precision
np.set_printoptions(precision = 6, suppress = True)

start_time = time.time()

if part_num == 'a':
	part_a()
elif part_num == 'b':
	random.seed(0)
	part_b()
elif part_num == 'c':
	part_c()
elif part_num == 'd':
	part_d()
elif part_num == 'e':
	part_e()

end_time = time.time()
print("Time taken:", end_time - start_time)