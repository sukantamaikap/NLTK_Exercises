from nltk.corpus import wordnet

synset = wordnet.synsets('plan')

#all
print(synset)

# lemma
print(synset[0].lemmas()[0].name)

#def
print(synset[0].definition())

#examples
print(synset[1].examples())

#synonyms & antonyms
synonyms = []
antonyms = []

print(wordnet.synsets('good'))
for word in wordnet.synsets('good'):
        for l in word.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

print(synonyms)
print(antonyms)

#similarities
word1 = wordnet.synset('dusk.n.01')
word2 = wordnet.synset('sunset.n.01')

print(word1.wup_similarity(word2))