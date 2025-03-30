import scrapy
import re

class CodeSpider(scrapy.Spider):
    name = "codespider"
    start_urls = ["https://huggingface.co/tasks/text-generation"]  # Replace with the actual site

    def clean_code(self, code):
        # Example of cleaning code (removing non-alphanumeric characters)
        return re.sub(r"[^a-zA-Z0-9_]", "", code)  # Simple example of cleaning

    def parse(self, response):
        # Extract code snippets and clean them
        for code_block in response.css("code"):
            raw_code = code_block.get()
            cleaned_code = self.clean_code(raw_code)  # Process and clean the code

            yield {
                "code": cleaned_code,
                "language": response.css("meta[language]::attr(content)").get(),
                "title": response.css("h1::text").get(),
            }

        # Handle pagination (move to next page if available)
        next_page = response.css("a.next::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse)
