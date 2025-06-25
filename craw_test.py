from firecrawl import FirecrawlApp

app = FirecrawlApp(
  api_key="smth",
  api_url='http://localhost:3002'
)

# Crawl a website:
# url = 'https://archives.nyobserver.com/2022/03/07/war-is-bad-for-culture-not-least-of-all-because-it-turns-our-cultural-institutions-into-bastions-of-jingoism'
url = 'https://whitney.org/collection/works/65848'
crawl_result = app.scrape_url(
  url,
  params={
    'formats': ['markdown'],
  }
)

# crawl_result = app.crawl_url(
#   url,
#   params={
#     'limit': 1,
#     'scrapeOptions': {'formats': ['markdown']}
#   }
# )
print(crawl_result)