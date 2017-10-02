import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
from nltk.stem import WordNetLemmatizer

content = state_union.raw('2006-GWBush.txt')
tokenizer = PunktSentenceTokenizer()
tokenised = tokenizer.tokenize(content)

def process_content():
    try:
        for i in tokenised:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            print(tagged)

            # chunking and chinking
            chunk_gram = r'''Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{'''
            # }<put chinking content inside this>{ and {<chunking content inside this>}
            chunk_parser = nltk.RegexpParser(chunk_gram)
            chunked = chunk_parser.parse(tagged)
            print(chunked)
            #chunked.draw()


            # find named entity
            named_entity = nltk.ne_chunk(tagged, binary=True)
            #named_entity.draw()

            #word lemmatizing
            lemmatizer = WordNetLemmatizer()
            print(lemmatizer.lemmatize('cats'))
            print(lemmatizer.lemmatize('algae'))
            print(lemmatizer.lemmatize('better', pos='a'))


    except Exception as e:
        print(str(e))

process_content()