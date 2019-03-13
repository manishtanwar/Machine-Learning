import utils
import sys
import string
import random
import numpy as np
import math
import sklearn
import time

train_file = sys.argv[1]
test_file = sys.argv[2]
part_num = sys.argv[3]

#---------------------- Part (a) --------------------------

dict = dict()
class_word_count = np.zeros(5)
class_docu_count = np.zeros(5)
m_test = 0
correct_pred = 0

y_pred = []
y_actual = []

# Training:
def train():
	itr = utils.json_reader(train_file)
	m = 0
	for item in itr:
		m+=1

		stars = int(item["stars"]) - 1
		word_list = item["text"].lower().translate(str.maketrans('', '', string.punctuation)).split()
		# ----- debug -----
		if m == 100:
			aa = utils.getStemmedDocuments(item["text"])
			# print(item["text"])
			# print(aa)
			# print(word_list)
			break
		# -----------------

		# ----- debug -----
		# if m <= 3:
		# 	print(item["text"])
		# 	print(word_list)
		# -----------------

		
		class_word_count[stars] += len(word_list)
		class_docu_count[stars] += 1

		for word in word_list:
			if word in dict:
				dict[word][stars] += 1
			else:
				dict[word] = np.zeros(5)
				dict[word][stars] = 1

# Testing:
def test(stem_flag=False):
	itr = utils.json_reader(test_file)
	global m_test
	global correct_pred
	for item in itr:
		m_test += 1
		# ----- debug -----
		if m_test == 100:
			break
		# -----------------
		stars = int(item["stars"]) - 1
		word_list = item["text"].lower().translate(str.maketrans('', '', string.punctuation)).split()

		pred_stars = 0
		max_prob = -(1e9)

		for i in range(5):
		    # P(y)
			prob = math.log(class_docu_count[i])
			
			for word in word_list:
				count = dict[word][i] if word in dict else 0
	            # P(xj/y)
				prob += math.log((count + 1) / (class_word_count[i] + len(dict)))

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

#---------------------- Part (e) --------------------------
#---------------------- Part (g) --------------------------



start_time = time.time()

if part_num == 'a':
	part_a()
elif part_num == 'b':
	random.seed(0)
	part_b()
elif part_num == 'c':
	part_c()

end_time = time.time()
print(end_time - start_time)