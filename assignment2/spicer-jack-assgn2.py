from collections import Counter
from collections import defaultdict
from itertools import islice
import random
import bisect
import re
import numpy as np



#
#
# HELPER FUNCTIONS
#
#


#Takes in a filename and returns the number of words in the file
#The words are returned in a dictionary with the word as the key and the count as the value
def countWords(file_name):
	file = open(file_name)
	wordcounts = Counter(file.read().split())
	return wordcounts

#replaces all words in the sample text with unk
def replaceWithUnk(file_name):
	with open(file_name) as f:
		words = [word for line in f for word in line.split()]

	wordcounts = countWords(file_name)
	for i, word in enumerate(words):
		if(wordcounts[word] < 2):
			words[i] = "<unk>"

	return words

#Converts words with frequencies <= 1 to <unk>
def unkify(freqs):
	newFreqs = {"<unk>": 0}
	for word in freqs:
		if (freqs[word] > 1):
			newFreqs[word] = freqs[word]
		else:
			newFreqs["<unk>"] = newFreqs["<unk>"] + 1
	return newFreqs

#Computes the dot product of a list of probabilities.
#Modular enough to be used for unigrams and bigrams
def product(nums):
	prod = 1
	for num in nums: prod = prod * num
	return prod




#
#
# UNIGRAM MODEL
#
#


#Complete function that utilizes helpers and returns the sentence probability using a unigram model
def unigramModel(sentence, filename):
	preUnkWordcounts = countWords(filename)
	wordcounts = unkify(preUnkWordcounts)
	return sentenceProbUnigram(sentence.split(), wordcounts)

#Computes the probability of a sentence
def sentenceProbUnigram(sentence, freqs):
    return product(uniProb(freqs, word) for word in sentence)
       
#Takes in the list of wordcounts and a word
#Returns the probability of that word occuring
#Credit: In class jupyter notebook
def uniProb(freqs, word): 
	if word in freqs:
		return freqs[word]/sum(freqs.values())
	else:
		return freqs["<unk>"]/sum(freqs.values())



#
#
# BIGRAM MODEL
#
#



#Complete function the calculates the sentence probability using a bigram model

def bigramModel(sentence, filename):
	#get all unigram counts
	preUnkWordcounts = countWords(filename)
	unigramCounts = unkify(preUnkWordcounts)
	vocabSize = len(unigramCounts)

	#get all bigram counts
	wordsList = replaceWithUnk(filename)
	words = " ".join(wordsList)
	bigrams = re.findall('[<a-z>\-\'\/\_\.]+', words)
	bigramCounts = Counter(zip(bigrams,bigrams[1:]))
	# bigramCounts = Counter(zip(words, islice(words, 1, None)))

	#preprocess sentence
	for word in sentence.split():
		if word not in unigramCounts.keys():
			sentence = sentence.replace(word, "<unk>", 1)
	sentence = sentence.split()

	#calculate probability
	nums = []
	for i in range(0,len(sentence)-1):
		nums.append(biProb(bigramCounts, unigramCounts, sentence[i+1], sentence[i], vocabSize))
	return product(nums)


#calculate probability of P(word1|word2) = P(word2, word1)/ P(word2)
def biProb(bigramCounts, unigramCounts, word1, word2, vocabSize):
	if (word2, word1) in bigramCounts:
		#add-1 smoothing
		return (bigramCounts[(word2, word1)] + 1)/(unigramCounts[word2] + vocabSize-1)
	else:
		#unknown word, smooth it
		return 1/(unigramCounts[word2] + vocabSize)




#
#
# SENTENCE GENERATION
#
#




def shannonMethod(filename):
	#get all unigram counts
	preUnkWordcounts = countWords(filename)
	unigramCounts = unkify(preUnkWordcounts)

	#get all bigram counts
	wordsList = replaceWithUnk(filename)
	words = " ".join(wordsList)
	bigrams = re.findall('[<a-z>\-\'\/\_\.]+', words)
	bigramCounts = Counter(zip(bigrams,bigrams[1:]))

	#start building the sentence
	currentWord = "<s>"
	sentence = ["<s>"]
	count = 0

	#stop if we pass 100 words or if we reach the end of sentence marker
	while(currentWord != "</s>" and count < 100):
		#create a probability distribution for the next word
		bins = []
		wordChoices = []
		binCounter = 0
		for bigram in bigramCounts.keys():
			if bigram[0] == currentWord:
				#probablity of the next word given the current word = occurences of next word given current word/all instances of current word
				binCounter = binCounter + (bigramCounts[bigram])/(unigramCounts[currentWord])
				bins.append(binCounter)
				wordChoices.append(bigram[1])

		#choose a random index, pick it out of the bins, add it to the sentence, find next word
		rndIndex = random.random()
		nextWord = wordChoices[bisect.bisect(bins, rndIndex)]
		if(nextWord != "<unk>" and nextWord != "<s>"):
			sentence.append(nextWord)
			currentWord = nextWord
			count += 1
		else:
			continue
		
	
	return " ".join(sentence)

def main():

	# print(unigramModel("<s> i want chinese food </s>","berp-training.txt"))
	# print(unigramModel("<s> tell me about pasta jays </s>","berp-training.txt"))
	# print(bigramModel("<s> i want chinese food </s>", "berp-training.txt"))
	# print(bigramModel("<s> tell me about pasta jays </s>","berp-training.txt"))


	# unigramProbs = []
	# bigramProbs = []
	# with open('berp-100-test.txt') as f:
	# 	lines = f.readlines()
	# 	for line in lines:
	# 		unigramProbs.append(unigramModel(line, "berp-training.txt"))
	# 		bigramProbs.append(bigramModel(line, "berp-training.txt"))

	# with open('spicer-jack-assgn2-unigram-out.txt', 'w') as f:
	#     for item in unigramProbs:
	#         f.write("%s\n" % item)
	# with open('spicer-jack-assgn2-bigram-out.txt', 'w') as f:
	#     for item in bigramProbs:
	#         f.write("%s\n" % item)

	shannonSentences = []
	for i in range(0,100):
		shannonSentences.append(shannonMethod("berp-training.txt"))


	with open('spicer-jack-assgn2-bigram-rand-corpus.txt', 'w') as f:
	    for item in shannonSentences:
	        f.write("%s\n" % item)

	# print(shannonMethod("berp-training.txt"))






if __name__ == '__main__':
	main()