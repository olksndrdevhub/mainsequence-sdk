from polygon import RESTClient
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

import dotenv
dotenv.load_dotenv('../../../.env')

client = RESTClient(os.environ["POLYGON_API_KEY"])

# Set the date range
start_date = "2024-06-01"
end_date = "2024-09-04"

# Fetch news and extract sentiment for 'CRWD'
sentiment_count = []
for day in pd.date_range(start=start_date, end=end_date):
    daily_news = list(client.list_ticker_news("CRWD", published_utc=day.strftime("%Y-%m-%d"), limit=100))
    daily_sentiment = {
        'date': day.strftime("%Y-%m-%d"),
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    for article in daily_news:
        if hasattr(article, 'insights') and article.insights:
            for insight in article.insights:
                if insight.sentiment == 'positive':
                    daily_sentiment['positive'] += 1
                elif insight.sentiment == 'negative':
                    daily_sentiment['negative'] += 1
                elif insight.sentiment == 'neutral':
                    daily_sentiment['neutral'] += 1
    sentiment_count.append(daily_sentiment)

# Convert to DataFramesentiment_count
df_sentiment = pd.DataFrame(sentiment_count)

# Convert 'date' column to datetime
df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])

# Set the date as the index
df_sentiment.set_index('date', inplace=True)

# Plotting the data
plt.figure(figsize=(20, 10))
plt.plot(df_sentiment['positive'], label='Positive', color='green')
plt.plot(df_sentiment['negative'], label='Negative', color='red')
plt.plot(df_sentiment['neutral'], label='Neutral', color='grey', linestyle='--')
plt.title('Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)

# Format the x-axis to display dates better
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Adjust interval as needed
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()  # Rotation

# Saving the plot as an image file
plt.savefig('sentiment_over_time.png')
plt.show()