import time
import numpy as np
import pandas as pd
from newspaper import Article
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# searching articles by keywords
search_keywords = [
 'Koroona',
 'Koroonapandeemia',
 'COVID',
 'COVID-19',
 'Vaktsiin',
 'Pandeemia',
 'Immuunsus',
 'Tervis',
 'Arst',
 'Ravi',
 'Rahvatervis',
 'Nakatumine',
 'Nakkav',
 'Tõhustusdoos',
 'Vaktsineerimine',
 'mRNA',
 'Kõrvaltoime',
 'Surm',
 'Karantiin',
 'Distantseerumine',
 'Epidemioloogia',
 'Mask',
 'Homeöpaatia',
 'Meditsiin',
 'Meedik',
 'Haigus',
 'Viirus',
 'Vähk',
 'Põletik'
 ]

# sources from FB
sites = [
    'https://objektiiv.ee/rubriik/arvamus/page/',
    'https://objektiiv.ee/rubriik/uudised/page/',
    'https://uueduudised.ee/rubriik/arvamus/page/',
    'https://uueduudised.ee/rubriik/uudis/eesti/page/',
    'https://uueduudised.ee/rubriik/uudis/maailm/page/',
    'https://tervise.geenius.ee/koik-lood/lk/',
    'https://www.telegram.ee/category/teadus-ja-tulevik/page/',
    'https://www.telegram.ee/category/toit-ja-tervis/page/',
    'https://www.telegram.ee/category/maailm/page/',
    'https://www.telegram.ee/category/eesti/page/',
    'https://www.telegram.ee/category/arvamus/page/',
    'https://www.telegram.ee/category/nwo/page/',
    'https://www.delfi.ee/kategooria/120/eesti?page=',
    'https://www.delfi.ee/kategooria/123/maailm?page=',
    'https://epl.delfi.ee/kategooria/67583634/arvamus?page=',
    'https://epl.delfi.ee/kategooria/67583608/uudised?page='
]

def get_urls_and_filter(base_url, start_page, end_page, keywords):
    filtered_urls = []
    # start_time = time.time()

    for page_number in range(start_page, end_page + 1):
        # elapsed_time = time.time() - start_time
        # limit = 120
        # if elapsed_time > limit * 60:
        #     print("Time limit exceeded. Stopping the fetching process.")
        #     break

        url = f"{base_url}{page_number}/"
        response = requests.get(url)

        if response.status_code == 404:
            print(f"Error fetching page {page_number}. Status code: {response.status_code}. Stopping the process.")
            break

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            if base_url.startswith("https://objektiiv.ee"):
                links = [a['href'] for a in soup.select('h4.card-title a[href]')]
            elif base_url.startswith("https://uueduudised.ee"):
                items = soup.find_all('div', class_='col-12 fw-bold pt-0 uu-sm-img-title')
                links = [item.find('a', class_='uu-list-title')['href'] for item in items if item.find('a', class_='uu-list-title')]
            elif base_url.startswith("https://tervise.geenius.ee"):
                links = [a['href'] for a in soup.select('a.link-unstyled.link-hover')]
            elif base_url.startswith("https://www.telegram.ee"):
                items = soup.find_all('div', class_='grid-item')
                links = [item.find('a', href=True)['href'] for item in items if item.find('a', href=True)]
            elif base_url.startswith("https://www.delfi.ee") or base_url.startswith("https://epl.delfi.ee"):
                items = soup.find_all('h5', class_='C-headline-title')
                links = [urljoin(url, item.find('a', href=True)['href']) for item in items if item.find('a', href=True)]
            else:
                print("Unsupported base URL format.")
                return filtered_urls

            for link in links:
                try:
                    article = Article(link)
                    article.download()
                    article.parse()
                    article_text = article.text

                    if any(keyword.lower() in article_text.lower() for keyword in keywords):
                        filtered_urls.append(link)

                except Exception as e:
                    print(f"Error processing article {link}: {str(e)}")
                    continue
                
                time.sleep(1) 
        else:
            print(f"Error fetching page {page_number}. Status code: {response.status_code}")

    return filtered_urls

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def clean_final_url_list(lst):
    unique_values = list(set(lst))
    cleaned_list = [value for value in unique_values if not (isinstance(value, float) and np.isnan(value))]
    return cleaned_list

# urls = []
# start_page = 1
# end_page = 500
# minutes = 8 * 60

# start_time = time.time() 

# for site in sites:
#     print(f'Starting the fetching for: {site}')
#     result_urls = get_urls_and_filter(site, start_page, end_page, search_keywords)
#     urls.append(result_urls)

#     elapsed_time = time.time() - start_time

#     if elapsed_time >= minutes * 60: 
#         print(f"{minutes} minutes have passed. Exiting the loop.")
#         break

#     time.sleep(1) 

# clean_urls = clean_final_url_list(flatten(urls))
# urls_df = pd.DataFrame(clean_urls, columns=["URL"])
# urls_df.to_csv('est-news-urls.csv', index=True)