import feedparser
import openai
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set OpenAI API Key (Ensure it's stored securely)
OPENAI_API_KEY = "Put API Key Here"  # Replace with your actual key
if not OPENAI_API_KEY:
    raise ValueError("Please set your OpenAI API key!")

# Fetch News from Financial Sources
def fetch_news():
    sources = {
        "Bloomberg": "https://www.bloomberg.com/feed",
        "WSJ": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
        "Seeking Alpha": "https://seekingalpha.com/feed.xml",
        "Morningstar": "https://www.morningstar.com/rss"
    }
    
    articles = []
    for source, url in sources.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:  # Get top 3 articles per source
            articles.append({
                "Source": source,
                "Title": entry.title,
                "Link": entry.link,
                "Summary": entry.get("summary", "No summary available")  # Handle missing summaries
            })
    return articles

# AI-Powered Summarization
def summarize_article(article_text):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",  # Use gpt-3.5-turbo if hitting API limits
            messages=[
                {"role": "system", "content": "Summarize this financial news article in 2-3 sentences."},
                {"role": "user", "content": article_text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Process and Summarize News
def generate_summaries():
    news_articles = fetch_news()
    summarized_articles = []

    for article in news_articles:
        summary = summarize_article(article["Summary"])
        summarized_articles.append({
            "Source": article["Source"],
            "Title": article["Title"],
            "Summary": summary,
            "Link": article["Link"]
        })
    
    return pd.DataFrame(summarized_articles)

# Display News Summaries
def display_news(df):
    print("\n📰 **Financial News Summaries:**\n")
    print(df.to_string(index=False))  # Display full DataFrame in terminal

# Plot News Sources
def plot_news_sources(df):
    plt.figure(figsize=(8,5))
    df["Source"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Number of Articles per News Source", fontsize=14)
    plt.xlabel("News Source")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45)
    plt.show()

# Save to CSV
def save_to_csv(df):
    df.to_csv("financial_news_summaries.csv", index=False)
    print("\n✅ Data saved to `financial_news_summaries.csv`")

# Main Function
if __name__ == "__main__":
    df_summaries = generate_summaries()
    display_news(df_summaries)
    save_to_csv(df_summaries)
    plot_news_sources(df_summaries)  # Visualize article sources