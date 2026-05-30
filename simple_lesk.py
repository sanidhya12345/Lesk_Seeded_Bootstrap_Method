from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')  
sent="I went to the bank to deposit money."

tokens=word_tokenize(sent)
syn=lesk(tokens,'bank','n')
print(syn,syn.definition())
