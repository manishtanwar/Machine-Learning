import utils
import sys
import string
import numpy as np

train_file = sys.argv[1]
test_file = sys.argv[2]
part_num = sys.argv[3]

dict_list = [dict() for i in range(5)]
class_word_count = np.zeros(5)
class_docu_count = np.zeros(5)

# Training:
itr = utils.json_reader(train_file)
m = 0
for item in itr:
    m+=1
    # if(ind==2):
    #     break
    stars = int(item["stars"]) - 1
    word_list = item["text"].translate(str.maketrans('', '', string.punctuation)).split()
    class_word_count[stars] += len(word_list)
    class_docu_count[stars] += 1

    for word in word_list:
        if word in dict_list[stars]:
            dict_list[stars][word] += 1
        else:
            dict_list[stars][word] = 1

# Testing:
itr = utils.json_reader(test_file)
m_test = 0
correct_pred = 0

for item in itr:
    m_test += 1
    stars = int(item["stars"]) - 1
    word_list = item["text"].translate(str.maketrans('', '', string.punctuation)).split()
    
    pred_stars = 0
    max_prob = 0.

    for i in range(5):
        # P(y)
        prob = math.log(class_docu_count)
        
        for word in word_list:
            # P(xj/y)
            prob += math.log((dict_list[i][word] + 1) / ())


accuracy = correct_pred / m_test
printf("accuracy : %f",accuracy)