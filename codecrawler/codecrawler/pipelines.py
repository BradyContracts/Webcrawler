# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

import sqlite3

class SQLitePipeline:
    def open_spider(self, spider):
        self.conn = sqlite3.connect("scraped_data.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_data (
                id INTEGER PRIMARY KEY,
                code TEXT,
                language TEXT,
                title TEXT
            )
        ''')

    def close_spider(self, spider):
        self.conn.commit()
        self.conn.close()

    def process_item(self, item, spider):
        self.cursor.execute('''
            INSERT INTO code_data (code, language, title)
            VALUES (?, ?, ?)
        ''', (item["code"], item["language"], item["title"]))
        return item


class CodecrawlerPipeline:
    def process_item(self, item, spider):
        return item
import re

class CleanCodePipeline:
    def process_item(self, item, spider):
        # Clean the 'code' field by removing any non-alphanumeric characters
        if 'code' in item:
            item['code'] = self.clean_code(item['code'])
        return item

    def clean_code(self, code):
        # Example cleaning function to remove non-alphanumeric characters
        return re.sub(r"[^a-zA-Z0-9_]", "", code)  # This can be adjusted for more complex cleaning

    


