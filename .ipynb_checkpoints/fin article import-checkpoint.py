import feedparser

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
        for entry in feed.entries[:5]:  # Fetch top 5 articles per source
            articles.append({
                "source": source,
                "title": entry.title,
                "link": entry.link,
                "summary": entry.summary
            })
    return articles