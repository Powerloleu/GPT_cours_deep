# We have a few errors in the test that we're still working on

from bloomfilter import BloomFilter
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

print('Getting dataset...')
imdb_dataset = pd.read_csv('IMDB_Dataset.csv')

# Sample data for the IMDb dataset
data_exemple = {
    'review': [
        "This movie is amazing! The acting is superb.",
        "I didn't like the plot, but the cinematography was great.",
        "An absolute masterpiece.",
        "The storyline was confusing, and the characters were not well-developed.",
        "A must-watch for any film enthusiast.",
        "I couldn't follow the narrative, but the visuals were stunning.",
        "One of the worst movies I've seen.",
        "The dialogue felt forced, and the pacing was off.",
        "Highly recommended. The twists kept me on the edge of my seat.",
        "Not my cup of tea. The ending was disappointing."
    ]
}

imdb_dataset = pd.DataFrame(data_exemple)

print('--- Getting nltk data...')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("--- Downloading punkt...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("--- Downloading stopwords...")
    nltk.download('stopwords')

stpw = stopwords.words('english')

def treat_data(text):
    stpw_and_punct = stpw + list(string.punctuation)
    tokens = [word for word in word_tokenize(text.lower()) if word not in stpw_and_punct]
    return pd.Series([len(tokens), ' '.join(tokens)], index=['token_count', 'treated_review'])

print('--- Treating dataset...')
imdb_dataset = imdb_dataset.iloc[:10]
imdb_dataset[['token_count', 'treated_review']] = imdb_dataset['review'].apply(treat_data)
imdb_train, imdb_test = train_test_split(imdb_dataset, test_size=0.2)
x_train = imdb_train['treated_review']
x_test = imdb_test['treated_review']

print('--- Creating Bloom filter...')
items_count = imdb_test['token_count'].sum()
fp_prob = 1e-20
bloom_filter = BloomFilter(items_count = items_count, fp_prob = fp_prob)

print(f'items count: {items_count}')
print(f'Bloom filter size: {bloom_filter.size}')
print(f'Bloom hash functions: {bloom_filter.hash_count}')

print('--- Adding 8-grams to the Bloom filter') 
_ = [bloom_filter.add(ngram) for review in x_train for ngram in [review[i:i+8] for i in range(len(review)-7)]]

print('--- Checking all 8-grams in the test set')
matching_count = sum(imdb_test['treated_review']
                     .apply(lambda review: sum(bloom_filter.check(ngram) 
                                               for ngram in [review[i:i+8] for i in range(len(review)-7)])))

percentage_matched = (matching_count / items_count) * 100
#print(f"Percentage of 8-grams found in the training set: {percentage_matched:.2f}%")
#print(f"Bloom count of 8-grams found in the training set: {matching_count}")
