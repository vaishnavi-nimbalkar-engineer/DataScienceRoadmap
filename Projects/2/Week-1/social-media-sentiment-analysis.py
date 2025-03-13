import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Sample Twitter data (in a real project, you would use Twitter API or a dataset)
# Creating a synthetic dataset for this example
topics = ['climate change', 'artificial intelligence', 'politics', 'healthcare', 'economy']
sentiments = ['positive', 'negative', 'neutral']
weights = [0.3, 0.4, 0.3]  # Probability weights for sentiments

np.random.seed(42)
n_tweets = 1000

# Generate synthetic tweets
tweets = []
for _ in range(n_tweets):
    topic = np.random.choice(topics)
    sentiment_category = np.random.choice(sentiments, p=weights)
    
    # Create tweet text based on sentiment and topic
    if sentiment_category == 'positive':
        sentiment_words = np.random.choice(['good', 'great', 'excellent', 'amazing', 'love', 'impressive'], size=2)
        tweet_text = f"I think {topic} is {sentiment_words[0]} and {sentiment_words[1]}! #excited"
    elif sentiment_category == 'negative':
        sentiment_words = np.random.choice(['bad', 'terrible', 'awful', 'concerning', 'hate', 'disappointing'], size=2)
        tweet_text = f"I think {topic} is {sentiment_words[0]} and {sentiment_words[1]}. #worried"
    else:  # neutral
        tweet_text = f"Here's some information about {topic}. #facts"
    
    # Add some randomness to tweet length and structure
    if np.random.random() > 0.5:
        tweet_text = f"{tweet_text} This is what I've been thinking about lately."
    
    tweet_date = pd.to_datetime('2023-01-01') + pd.Timedelta(days=int(np.random.random() * 100))
    
    tweets.append({
        'text': tweet_text,
        'topic': topic,
        'true_sentiment': sentiment_category,
        'date': tweet_date
    })

tweets_df = pd.DataFrame(tweets)

# Clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    text = text.lower()                  # Convert to lowercase
    return text

tweets_df['clean_text'] = tweets_df['text'].apply(clean_text)

# Perform sentiment analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

tweets_df['polarity'] = tweets_df['clean_text'].apply(get_polarity)
tweets_df['predicted_sentiment'] = tweets_df['clean_text'].apply(get_sentiment)

# Analyze results
sentiment_counts = tweets_df['predicted_sentiment'].value_counts()
print("Sentiment Distribution:")
print(sentiment_counts)

# Calculate accuracy (how well did TextBlob match our synthetic labels)
accuracy = (tweets_df['predicted_sentiment'] == tweets_df['true_sentiment']).mean()
print(f"\nSentiment Analysis Accuracy: {accuracy:.2f}")

# Sentiment by topic
topic_sentiment = tweets_df.groupby('topic')['polarity'].mean().sort_values()
print("\nAverage Sentiment by Topic:")
print(topic_sentiment)

# Visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='predicted_sentiment', data=tweets_df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()

# Time series analysis
tweets_df['date'] = pd.to_datetime(tweets_df['date'])
tweets_df['day'] = tweets_df['date'].dt.date
daily_sentiment = tweets_df.groupby('day')['polarity'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(daily_sentiment['day'], daily_sentiment['polarity'])
plt.title('Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Polarity')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Word frequency analysis
def get_word_freq(texts):
    all_words = ' '.join(texts).split()
    return Counter(all_words)

positive_words = get_word_freq(tweets_df[tweets_df['predicted_sentiment'] == 'positive']['clean_text'])
negative_words = get_word_freq(tweets_df[tweets_df['predicted_sentiment'] == 'negative']['clean_text'])

print("\nMost Common Positive Words:")
print(pd.DataFrame(positive_words.most_common(10), columns=['Word', 'Count']))

print("\nMost Common Negative Words:")
print(pd.DataFrame(negative_words.most_common(10), columns=['Word', 'Count']))
