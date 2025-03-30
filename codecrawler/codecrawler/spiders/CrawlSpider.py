import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class DynamicSpider(CrawlSpider):
    name = "dynamic_spider"
    allowed_domains = ['example.com']
    start_urls = ['https://example.com/start_page']

    rules = (
        Rule(LinkExtractor(allow=('/questions/', '/repos/')), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        # Parse each page from the dynamically discovered links
        pass
