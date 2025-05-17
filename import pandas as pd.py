import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

data = {
    'timestamp': pd.to_datetime(['2025-05-10 08:00:00', '2025-05-10 10:30:00', '2025-05-11 12:00:00',
                                   '2025-05-12 15:45:00', '2025-05-13 09:15:00', '2025-05-13 18:00:00',
                                   '2025-05-14 11:00:00', '2025-05-15 14:30:00', '2025-05-16 07:45:00',
                                   '2025-05-17 16:00:00']),
    'source': ['Twitter', 'Forum', 'News Comment', 'Twitter', 'Survey',
               'Forum', 'Twitter', 'News Comment', 'Survey', 'Forum'],
    'text': [
        "This new fare system is confusing and expensive!",
        "Finally, a fairer way to pay for transport based on distance.",
        "I think the flat rate was much simpler.",
        "The cost for long journeys has significantly increased. Unfair!",
        "Generally satisfied with the change, seems more equitable.",
        "Is there a clear explanation of how the fares are calculated?",
        "Heard some people are being overcharged. Need clarification.",
        "The app is difficult to use for calculating fares.",
        "Neutral on the pricing, but the app needs improvement.",
        "More transparency is needed regarding the pricing structure."
    ]
}
df = pd.DataFrame(data)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['text'].apply(analyze_sentiment)

df['date'] = df['timestamp'].dt.date
sentiment_over_time = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
sentiment_over_time['total'] = sentiment_over_time['Positive'] + sentiment_over_time['Negative'] + sentiment_over_time['Neutral']
sentiment_over_time['positive_ratio'] = sentiment_over_time['Positive'] / sentiment_over_time['total']
sentiment_over_time['negative_ratio'] = sentiment_over_time['Negative'] / sentiment_over_time['total']

sentiment_by_source = df.groupby('source')['sentiment'].value_counts(normalize=True).unstack(fill_value=0)

plt.figure(figsize=(12, 6))
plt.plot(sentiment_over_time.index, sentiment_over_time['positive_ratio'], label='Positive')
plt.plot(sentiment_over_time.index, sentiment_over_time['negative_ratio'], label='Negative')
plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Sentiment Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sentiment_by_source.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sentiment Distribution by Source')
plt.xlabel('Data Source')
plt.ylabel('Proportion of Sentiment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

negative_spike_threshold = 0.6
recent_negative_ratio = sentiment_over_time['negative_ratio'].iloc[-1]

print("\n--- Issue Flagging and Recommendations ---")
if recent_negative_ratio > negative_spike_threshold:
    print(f"Warning: High negative sentiment detected on {sentiment_over_time.index[-1]} (Negative Ratio: {recent_negative_ratio:.2f})")
    print("- Investigate the reasons behind this recent negative spike. Check specific comments and topics.")

print("\nSentiment by Source:")
print(sentiment_by_source)
if 'Twitter' in sentiment_by_source and sentiment_by_source.get('Twitter', {}).get('Negative', 0) > 0.5:
    print("- High negative sentiment observed on Twitter. Consider targeted communication on this platform.")
if 'Forum' in sentiment_by_source and sentiment_by_source.get('Forum', {}).get('Neutral', 0) > 0.4:
    print("- Significant neutral sentiment on Forums. Explore these discussions for specific concerns or ambiguities.")
if 'News Comment' in sentiment_by_source and sentiment_by_source.get('News Comment', {}).get('Negative', 0) > 0.4:
    print("- Negative sentiment prevalent in News Comments. Analyze these comments for specific grievances.")

print("\nGeneral Recommendations:")
print("- Implement multilingual sentiment analysis to accurately capture feedback in Kinyarwanda and French.")
print("- Conduct topic modeling to identify the key themes and concerns driving public sentiment.")
print("- Analyze the sentiment around specific aspects of the new system (e.g., pricing for short vs. long distances, app usability).")
print("- Develop a dashboard to provide policymakers with an interactive overview of sentiment trends and key insights.")
print("- Establish clear channels for public feedback and actively address the concerns raised.")