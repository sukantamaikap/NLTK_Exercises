import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union

content = state_union.raw('2006-GWBush.txt')
tokenizer = PunktSentenceTokenizer()
tokenised = tokenizer.tokenize(content)

def process_content():
    try:
        for i in tokenised:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            print(tagged)

            # chunking
            chunk_gram = r'''Chunk: {<RB.?>*<NNP>+<NN>?}'''
            chunk_parser = nltk.RegexpParser(chunk_gram)
            chunked = chunk_parser.parse(tagged)

            print(chunked)
            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()