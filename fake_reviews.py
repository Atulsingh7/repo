import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from textblob import TextBlob

# Sample Data
data = {'review': ['This is a great restaurant', 'The food was awful', 'Absolutely loved the place', 
                   'Not worth the price', 'Fantastic service'], 
        'label': ['Genuine', 'Fake', 'Genuine', 'Fake', 'Genuine']}

df = pd.DataFrame(data)

# Preprocessing reviews for topic modeling
def preprocess_reviews(reviews):
    return [[word for word in review.lower().split()] for review in reviews]

# Generate topics using LDA
def topic_modeling(reviews, num_topics=2):
    dictionary = Dictionary(reviews)
    corpus = [dictionary.doc2bow(review) for review in reviews]
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
    topics = lda_model.print_topics(num_topics=num_topics)
    return topics

# Sentiment analysis
def sentiment_analysis(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity

# Prepare data for modeling
df['processed_review'] = preprocess_reviews(df['review'])
df['sentiment'] = df['review'].apply(sentiment_analysis)

# Output the topics
topics = topic_modeling(df['processed_review'])
print("Detected Topics:", topics)
print(df)
