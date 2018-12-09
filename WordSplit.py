from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import word2vec
import re
# nltk.download('punkt')

# corpus 1
# corpus = ['I like NLP', 'I like natural language processing', 'I like deep learning', 'I enjoy NLP']
# word_split = []
# for text in corpus:
#     word_split.append(word_tokenize(text))

# corpus 2
# with open('./corpus/gone_with_the_wind.txt', mode='r') as f:
#     document = f.readlines()
#     word_split = []
#     for text in document:
#         text = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ", text)
#         word_split.append(word_tokenize(text))

with open('./corpus/gone_with_the_wind.txt', mode='r') as f:
    document = f.read()
    sent_split = sent_tokenize(document)
    word_split = []
    for text in sent_split:
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ", text)
        word_split.append(word_tokenize(text))

print(word_split)
model = word2vec.Word2Vec(word_split, min_count=1, size=32)
# model.train([sent_split], total_examples=model.corpus_count, epochs=model.iter)
#
# save the model
model.save('./corpus/MyModel')
# model.wv.save_word2vec_format('./corpus/mymodel.txt', binary=False)
# model.wv.save_word2vec_format('./corpus/mymodel.bin.gz', binary=True)
