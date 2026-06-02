from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('semcor')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
sent="I went to the bank to deposit money."

tokens=word_tokenize(sent)
syn=lesk(tokens,'bank','n')
print(syn,syn.definition())
