from gensim.models import word2vec

model = word2vec.Word2Vec.load('./corpus/MyModel')
# model = KeyedVectors.load_word2vec_format('./corpus/mymodel.txt', binary=False)
# model = KeyedVectors.load_word2vec_format('./corpus/mymodel.bin.gz', binary=True)

print(model.wv.most_similar('Scarlett'))
print(model.wv.similarity('war', 'peace'))
