import re

class CodeProcessingPipeline:
    def process_item(self, item, spider):
        # Clean the code (remove unwanted tags, extra spaces, etc.)
        clean_code = re.sub(r'<.*?>', '', item['code'])  # Removes HTML tags
        item['code'] = clean_code.strip()
        
        # Optionally, handle other processing steps (e.g., language detection, formatting)
        
        return item
