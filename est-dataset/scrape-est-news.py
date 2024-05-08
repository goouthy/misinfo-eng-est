import pandas as pd
from newspaper import Article

def scrape_articles(urls, min_text_length=100, output_csv='news_data3.csv'):
    df = pd.DataFrame(columns=['URL', 'Canonical_Link', 'Text', 'Publish_Date', 'Authors', 'Tags', 'Meta_Lang'])
    for url in urls:
        try:
            article = Article(url, language='et')
            article.download()
            article.parse()
            df = df.append({
                'URL': url,
                'Canonical_Link': article.canonical_link,
                'Text': article.text,
                'Publish_Date': article.publish_date,
                'Authors': article.authors,
                'Tags': article.tags,
                'Meta_Lang': article.meta_lang
            }, ignore_index=True)
        except Exception as e:
            print(f"Error processing URL '{url}': {e}")
    df_filtered = df[(df['Text'] != '') & (df['Text'].str.len() >= min_text_length)]
    df_filtered.to_csv(output_csv, index=False)
    return df_filtered