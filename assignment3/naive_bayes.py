from collections import Counter
import string, math


#Develops unigram model given file of sentences of training data
def countWords(file_name):
	file = open(file_name)
	word_counts = file.read().split()
	new_word_count = []
	for word in word_counts:
		if len(word) > 2:
			if word[:2] == "ID":
				continue
			else: 
				word = word.lower()
				word = word.translate(str.maketrans('','',string.punctuation))
				new_word_count.append(word)
		else:
			word = word.lower()
			word = word.translate(str.maketrans('','',string.punctuation))
			new_word_count.append(word)
	return Counter(new_word_count)


#Returns either NEG or POS based off of a Naive Bayes model and an input sentence
def compute_class_prob(sentence, pos_counts, neg_counts):
	pos_vocab_size = len(pos_counts)
	neg_vocab_size = len(neg_counts)
	pos_prob = 0
	neg_prob = 0

	for word in sentence:
		if word in pos_counts:
			pos_prob = pos_prob + math.log((pos_counts[word]+1)/(sum(pos_counts.values())+pos_vocab_size))
		else:
			pos_prob = pos_prob + math.log((1)/(sum(pos_counts.values())+pos_vocab_size))


		if word in neg_counts:
			neg_prob = neg_prob + math.log((neg_counts[word]+1)/(sum(neg_counts.values())+neg_vocab_size))
		else:
			neg_prob = neg_prob + math.log((1)/(sum(neg_counts.values())+neg_vocab_size))


	if neg_prob > pos_prob:
		return "NEG"
	else:
		return "POS"



def main():
	pos_word_counts = countWords("hotelPosT-train.txt")
	neg_word_counts = countWords("hotelNegT-train.txt")
	neg_c = 0
	pos_c = 0
	output = []


	file_name = "HW3-testset.txt"
	with open(file_name) as f:
		lines = f.readlines()

	for sentence in lines:
		sentence_id = sentence.split()[0]
		sentence = sentence.lower()
		sentence = sentence.translate(str.maketrans('','',string.punctuation))
		sentence_list = sentence.split()[1:]
	
		result = compute_class_prob(sentence_list, pos_word_counts, neg_word_counts)

		output.append(sentence_id + "\t" + result + "\n")
	

	with open('Spicer-Jack-assgn3-out.txt', 'w') as f:
		for item in output:
			f.write(item)


if __name__ == '__main__':
	main()