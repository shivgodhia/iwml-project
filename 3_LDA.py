# %% imports
import pickle
from nltk.corpus import stopwords
from pprint import pprint
import datetime
import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# following https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#11createthedictionaryandcorpusneededfortopicmodeling
# %% Load data
with open('otter_reviews_filtered', 'rb') as file:
    reviews = pickle.load(file)
# extract just the text into one big list
data = [review['content'] for review in reviews]
print("Data loaded")
# %% text preprocessing
# removing punctuation, superfluous characters such as emojis and stopwords, and through lemmatization and tokenization

# prepare stopwords
stop_words = stopwords.words('english')
print("Prepared stopwords")
# tokenize words and clean up text
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))  
data_words = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

print("Built bigram and trigram models")
# Remove stopwords, make bigrams and lemmatize
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
print("Removed stopwords")

# Form trigrams
data_words_bigrams = make_trigrams(data_words_nostops)
print("Formed trigrams")

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print("Lemmatized data")

# %% Create dictionary and corpus needed for topic modelling

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
print("Created dictionary")

# Create Corpus
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_lemmatized]
print("Created corpus")

# Human readable format of corpus (term-frequency)
# [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

# %% Train or Load LDA, the topic model

# initialise variables
LOAD_LDA = True
# INPUT_MODEL_FILENAME = "lda_model_gensim_20topics-0103-1348"
INPUT_MODEL_FILENAME = "lda_model_mallet_47topics-0103-1547"

# if training, we need these
USE_MALLET_IMPLEMENTATION = True
NUM_TOPICS = 47
saved_results = [["NUM_TOPICS", "COHERENCE_SCORE", "MODEL_FILENAME"]]

OUTPUT_FILENAME = ""
# break the data down into topics, using Mallet's implementation via Gensim of Blei et al.'s Latent Dirichlet Allocation algorithm.
if LOAD_LDA:
    with open(INPUT_MODEL_FILENAME, 'rb') as file:
        lda_model = pickle.load(file)
        print("Loaded LDA model '{}'".format(INPUT_MODEL_FILENAME))
# Mallet's implementation via Gensim
elif USE_MALLET_IMPLEMENTATION:
    path_to_mallet_binary = "./mallet-2.0.8/bin/mallet"
    lda_model = LdaMallet(path_to_mallet_binary, corpus=corpus, num_topics=NUM_TOPICS, id2word=id2word)
    OUTPUT_FILENAME = 'lda_model_mallet_{}topics-{}'.format(NUM_TOPICS, datetime.datetime.now().strftime("%d%m-%H%M"))
    with open(OUTPUT_FILENAME,'wb') as file:
        pickle.dump(lda_model, file)
        print("LDA (mallet) complete and saved")
else:
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=NUM_TOPICS, 
                                        random_state=100,
                                        update_every=1,
                                        chunksize=100,
                                        passes=10,
                                        alpha='auto',
                                        per_word_topics=True)
    OUTPUT_FILENAME = 'lda_model_gensim_{}topics-{}'.format(NUM_TOPICS, datetime.datetime.now().strftime("%d%m-%H%M"))
    with open(OUTPUT_FILENAME,'wb') as file:
        pickle.dump(lda_model, file)
        print("LDA (gensim) complete and saved")


# %% analyse the LDA model's topics
# Print the Keyword in the topics
# pprint(lda_model.print_topics())

# Compute Perplexity
# print('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# saved_results.append([NUM_TOPICS, coherence_lda, OUTPUT_FILENAME])
# print('Coherence Score: ', coherence_lda)

# df = pd.DataFrame(saved_results)
# df.to_csv('lda_coherence_scores.csv', index=False, header=False)
# the main hyperparameter for the LDA model is the expected number of topics, testing 25 models with topic numbers from 2-100. Then choose the model w highest coherence score. 

# Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis
