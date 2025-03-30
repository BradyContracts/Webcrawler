import scrapy

class CodeSpider(scrapy.Spider):
    name = "codespider"
    start_urls = ["https://example.com"]  # Replace with the actual site

    def parse(self, response):
        # Extract code snippets
        for code_block in response.css("code"):
            yield {
                "code": code_block.get(),
                "language": response.css("meta[language]::attr(content)").get(),
                "title": response.css("h1::text").get(),
                # Add any other metadata you want to extract
            }

        # Handle pagination (move to next page if available)
        next_page = response.css("a.next::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse)
