from google_play_scraper import app, Sort, reviews_all
import pprint
import pickle
from collections import defaultdict

PACKAGE = 'com.aisense.otter'
LANG = 'en'
COUNTRY = 'us'
pp = pprint.PrettyPrinter()

# this gets some details about the app
# result = app(PACKAGE, LANG, COUNTRY)

""" Done already
reviews = reviews_all(PACKAGE, lang=LANG, country=COUNTRY, sort=Sort.NEWEST)

with open('otter_reviews','wb') as file:
    pickle.dump(reviews, file)
"""

# read with this:
with open('otter_reviews', 'rb') as file:
    reviews = pickle.load(file)

reviews_by_stars = defaultdict(list)

for review in reviews:
    reviews_by_stars[review['score']].append(review)

