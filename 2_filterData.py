# %% imports
import pickle
from collections import defaultdict
# %% Load data
with open('otter_reviews_with_sentiment', 'rb') as file:
    reviews = pickle.load(file)
    
# %% initialise variables
SCORE_THRESHOLD, MAGNITUDE_THRESHOLD = 0, 0.1
MIN_STARS, MAX_STARS = 1, 4
# %% filter based on sentiment and stars
def sentimentCriteriaMet(review,minMagnitude=MAGNITUDE_THRESHOLD, maxScore=SCORE_THRESHOLD):
    return review['sentimentAnnotations']['documentSentiment']['magnitude'] >= minMagnitude and review['sentimentAnnotations']['documentSentiment']['score'] <= maxScore

def starCriteriaMet(review, minStars=MIN_STARS, maxStars=MAX_STARS):
    return review['score'] in range(minStars, maxStars+1)
    
filteredAll = [review for review in reviews if sentimentCriteriaMet(review) and starCriteriaMet(review)]

filteredStar = [review for review in reviews if starCriteriaMet(review)]

filteredSentiment = [review for review in reviews if sentimentCriteriaMet(review)]

with open('otter_reviews_filtered','wb') as file:
    pickle.dump(filteredSentiment, file)
