import utils
import sys
import string
import numpy as np
import math

train_file = sys.argv[1]
test_file = sys.argv[2]
part_num = sys.argv[3]
dict = dict()

#---------------------- Part (a) --------------------------

class_word_count = np.zeros(5)
class_docu_count = np.zeros(5)
m_test = 0
correct_pred = 0

# Training:
# def train():
itr = utils.json_reader(train_file)
m = 0
for item in itr:
	m+=1
	stars = int(item["stars"]) - 1
	word_list = item["text"].translate(str.maketrans('', '', string.punctuation)).split()
	class_word_count[stars] += len(word_list)
	class_docu_count[stars] += 1

	for word in word_list:
		if word in dict:
			dict[word][stars] += 1
		else:
			dict[word] = np.zeros(5)
			dict[word][stars] = 1

# Testing:
# def test():
itr = utils.json_reader(test_file)

for item in itr:
	m_test += 1

	stars = int(item["stars"]) - 1
	word_list = item["text"].translate(str.maketrans('', '', string.punctuation)).split()
	
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
	if(pred_stars == stars):
		correct_pred+=1

# def part_a():
# 	test()
# 	train()
accuracy = correct_pred / m_test
print("accuracy :",accuracy)

#---------------------- Part (b) --------------------------



#---------------------- Part (c) --------------------------


# if part_num == 'a':
# 	part_a()