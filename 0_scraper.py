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


reviews = reviews_all(PACKAGE, lang=LANG, country=COUNTRY, sort=Sort.NEWEST)

with open('otter_reviews_scraped','wb') as file:
    pickle.dump(reviews, file)

