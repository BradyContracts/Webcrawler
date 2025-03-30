import re
from scrapy.utils.html import remove_tags

class CleanCodePipeline:
    def process_item(self, item, spider):
        # Clean the 'code' field by removing HTML tags and extra spaces
        if 'code' in item:
            item['code'] = self.clean_code(item['code'])
        return item

    def clean_code(self, code):
        # Remove HTML tags from the code
        cleaned_code = remove_tags(code)
        
        # Optionally, clean up extra spaces, newlines, or non-alphanumeric characters
        cleaned_code = re.sub(r'\s+', ' ', cleaned_code)  # Replace multiple spaces/newlines with a single space
        cleaned_code = re.sub(r"[^a-zA-Z0-9_\-(),.]+", " ", cleaned_code)  # Remove unwanted special characters (but keep basic punctuation)

        return cleaned_code.strip()  # Return cleaned code without leading/trailing spaces
