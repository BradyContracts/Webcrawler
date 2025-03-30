import scrapy
import json

class PythonOrgSpider(scrapy.Spider):
    name = "pythonorg_spider"
    
    start_urls = [
        "https://docs.python.org/3/tutorial/index.html"
    ]
    
    def parse(self, response):
        # Look for code blocks within the documentation
        code_blocks = response.xpath('//code/text()').getall()
        
        scraped_data = []
        for code in code_blocks:
            scraped_data.append({
                'site': 'python.org',
                'code': code.strip()
            })
        
        # Save data to a JSON file
        self.save_to_file(scraped_data)

    def save_to_file(self, data):
        with open('pythonorg_results.json', 'a') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.write("\n")

    