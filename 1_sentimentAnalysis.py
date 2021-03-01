# %% imports
import pickle
import json
from google.cloud import language_v1

# %% initialise variables
RESUME_INDEX = 0
# %% load the data
if RESUME_INDEX == 0:
    with open('otter_reviews_scraped', 'rb') as file:
        reviews = pickle.load(file)
elif RESUME_INDEX > 0:
    with open('otter_reviews_with_sentiment', 'rb') as file:
        reviews = pickle.load(file)
# %% get sentiment of all of them using Cloud Natural Language API
client = language_v1.LanguageServiceClient()
for i, review in enumerate(reviews):
    if i < RESUME_INDEX:
        continue
    reviewText = review["content"]
    document = language_v1.Document(content=reviewText, type_=language_v1.Document.Type.PLAIN_TEXT, language="en")
    annotations = client.analyze_sentiment(request={'document': document})
    # convert AnalyzeSentimentResponse class to dict and save it
    review["sentimentAnnotations"] = json.loads(annotations.__class__.to_json(annotations))
    reviews[i] = review
    # save every 100 if needed
    if i % 100 == 0:
        with open('otter_reviews_with_sentiment','wb') as file:
            pickle.dump(reviews, file)
    print("{} out of {} analysed".format(i, len(reviews)))



with open('otter_reviews_with_sentiment','wb') as file:
    pickle.dump(reviews, file)
