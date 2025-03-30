import scrapy

class MultiURLSpider(scrapy.Spider):
    name = "multi_url_spider"
    
    # List of URLs and their respective identifiers
    start_urls = [
        {"url": "https://github.com/username/repository", "site": "github"},
        {"url": "https://stackoverflow.com/questions/tagged/python", "site": "stackoverflow"},
        {"url": "https://www.geeksforgeeks.org/python-programming-language/", "site": "geeksforgeeks"},
        {"url": "https://codepen.io/collection/", "site": "codepen"},
    ]
    
    def start_requests(self):
        for site_info in self.start_urls:
            self.log(f"Scraping {site_info['site']} at {site_info['url']}")
            yield scrapy.Request(url=site_info["url"], callback=self.parse, meta={'site': site_info["site"]})
    
    def parse(self, response):
        site = response.meta['site']
        
        self.log(f"Parsing {site} site at {response.url}")
        
        if site == "github":
            yield from self.parse_github(response)
        elif site == "stackoverflow":
            yield from self.parse_stackoverflow(response)
        elif site == "geeksforgeeks":
            yield from self.parse_geeksforgeeks(response)
        elif site == "codepen":
            yield from self.parse_codepen(response)

    def parse_github(self, response):
        """Parse GitHub pages (repositories, issues, etc.)"""
        repo_name = response.url.split('/')[-1]
        code_blocks = response.xpath('//code//text()').getall()
        for code in code_blocks:
            yield {
                'site': 'github',
                'repo': repo_name,
                'code': code.strip()
            }

    def parse_stackoverflow(self, response):
        """Parse StackOverflow question pages"""
        questions = response.xpath('//div[@class="question-summary"]')
        for question in questions:
            title = question.xpath('.//h3/a/text()').get()
            link = question.xpath('.//h3/a/@href').get()
            question_code = question.xpath('.//div[@class="s-prose js-post-body"]/pre/code/text()').get()
            yield {
                'site': 'stackoverflow',
                'title': title.strip(),
                'url': f"https://stackoverflow.com{link}",
                'question_code': question_code.strip() if question_code else None
            }

    def parse_geeksforgeeks(self, response):
        """Parse GeeksforGeeks articles or tutorials"""
        article_title = response.xpath('//h1/text()').get()

        # Try capturing all <pre> and <code> tags to gather code
        article_code = response.xpath('//pre/text()').getall()

        # If no code found in <pre> tags, check for <code> tags in the content
        if not article_code:
            article_code = response.xpath('//code/text()').getall()

        # Clean up the extracted code (Remove extra spaces, newlines)
        code = "\n".join([line.strip() for line in article_code if line.strip()]).strip() if article_code else "No code found."

        yield {
            'site': 'geeksforgeeks',
            'article_title': article_title.strip() if article_title else None,
            'code': code
        }

    def parse_codepen(self, response):
        """Parse CodePen pages"""
        pens = response.xpath('//a[@class="PenPreview"]')
        for pen in pens:
            title = pen.xpath('.//h3/text()').get()
            link = pen.xpath('.//@href').get()
            yield {
                'site': 'codepen',
                'title': title.strip(),
                'url': link
            }
