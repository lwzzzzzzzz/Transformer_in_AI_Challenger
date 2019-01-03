import nltk
from nltk.tokenize import word_tokenize

a = "And then all along the colonnades there would've been statuary, including an image of Aeneas on this side, Romulus on this side, and the so-called summi viri, the great men of Rome, both Augustus' colleagues and also his rivals, in their portraits on either side: a kind of giant picture gallery, a giant portrait gallery of Rome, of the great men of Rome, of the greatest men of Rome, namely Augustus himself, and of his ancestry, both divine and mythological, via Aeneas and also Venus."
b = word_tokenize(a)
print(b)
print(len(b))