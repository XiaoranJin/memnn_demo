
from memnn import memnn
from sklearn import metrics
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import *
import numpy as np

TRAIN_FILE='data/en/qa2_two-supporting-facts_train.txt'
TEST_FILE='data/en/qa2_two-supporting-facts_test.txt'
D = 50
gamma = 0.1
alpha = 0.01

def readLines(filePath):
	output = []
	with open(filePath) as f:
		for line in f:
			ID, text = line.strip("\n").split(" ",1)
			ID = int(ID)
			text = text.replace(".","")

			if "\t" in text:
				sentence, answer, refs = text.split("\t")
				output.append({"id":ID, 
					"text":sentence.replace("?",""),
					"refs":map(int,refs.split()),
					"answer":answer,
					"type":"q"
					})

			else:
				output.append({"id":ID, "text":text,"type":"s"})
	return output

if __name__ == '__main__':
	train_lines, test_lines = readLines(TRAIN_FILE), readLines(TEST_FILE)

	lines = np.concatenate([train_lines, test_lines], axis=0)
	vectorizer = CountVectorizer()
	vectorizer.fit([x['text'] + ' ' + x['answer'] if 'answer' in x else x['text'] for x in lines])
	L = vectorizer.transform([x['text'] for x in lines]).toarray().astype(np.int32)
	L_train, L_test = L[xrange(len(train_lines))], L[xrange(len(train_lines),len(lines))]

	input_units = len(vectorizer.vocabulary_)
	model = memnn(input_units,D,gamma,alpha)

	R = np.diag(np.ones(input_units,dtype = np.int32))

	# Train memnn
	for epoch in range(5):
		total_cost = 0
		for i,line in enumerate(train_lines):
			if line['id'] == 1:
				start_index = i
			
			if i % 1000 == 0: print i

			if line['type'] == 'q':
				x_v = L_train[i]
				refs = line['refs']
				answer = line['answer']
				f1_v_index = refs[0] - 1 + start_index
				f2_v_index = refs[1] - 1 + start_index
				f1_v = L_train[f1_v_index]
				f2_v = L_train[f2_v_index]

				answer_index = vectorizer.vocabulary_[answer]
				r_v = R[answer_index]
				_r_v = R[[index for index in range(R.shape[0]) if index != answer_index]]

				memory_indice = [index for index in range(start_index,i) if train_lines[index]['type'] != "q"]
				memory_indice.remove(f1_v_index)
				_f1_v = L_train[memory_indice]

				memory_indice.remove(f2_v_index)
				_f2_v = L_train[memory_indice]

				if len(_f2_v) == 0: # all memory are facts
					_f2_v = _f1_v * 0

				total_cost += model.train(x_v, f1_v, _f1_v, f2_v, _f2_v, r_v, _r_v)[0]
				#print total_cost

		print "Epoch %d: " % (epoch + 1), total_cost


	# test memnn
	inverse_dict = dict([(value, key) for key,value in vectorizer.vocabulary_.iteritems()])
	total_cost = 0
	Y_true = []
	Y_pred = []

	for i,line in enumerate(test_lines):
		if line['id'] == 1:
			start_index = i

		if i % 1000 == 0: print i

		if line['type'] == 'q':
			x_v = L_test[i]

			refs = line['refs']
			answer = line['answer']
			Y_true.append(answer)

			memory_indice = [index for index in range(start_index,i) if test_lines[index]['type'] != "q"]
			memory_v_1 = L_test[memory_indice]

			r_predict = model.predict(x_v, memory_indice, memory_v_1, R)
			Y_pred.append(inverse_dict[r_predict])

	print metrics.classification_report(Y_true, Y_pred)
	print "Accuracy: ", sum([1 for x,y in zip(Y_true, Y_pred) if x == y])/float(len(Y_true))