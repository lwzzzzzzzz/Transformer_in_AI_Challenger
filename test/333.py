import jieba
import nltk

a = "价钱怎么样？—这儿的人还是实诚的"
b = jieba.lcut(a, cut_all=False, HMM=True)+["</S>"]

c = "a women in the house."
d = nltk.word_tokenize(c)
print(b, d)
print(b[:-1])