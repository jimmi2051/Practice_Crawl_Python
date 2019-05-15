import scrapy
import os
from .. import support 

class ThreadsSpider(scrapy.Spider):
    name = "threads"
    folder_path = "input"
    title_list = []
    page_default = "https://vnexpress.net/"
    url_list = [] 
    flag_save = False
    index_count = 1
    file_title = "output/title.txt"
    file_label = "output/label.txt"
    file_content = "output/content.txt"

    def write_file(self,content,file_path):
        with open(file_path,'a+') as file:
            file.write(content+",")

    def start_requests(self):
        
        input_title = open("input/title.txt", 'r+')
        data_title =input_title.readlines()
        for line in data_title:
            self.title_list.append(line)
        yield scrapy.Request(url=self.page_default,callback=self.getUrl)

    def getUrl(self,response):
        result = response.xpath('//*[@id="main_menu"]/a/@href').getall()
        titleToSearch = response.xpath('//*[@id="main_menu"]/a/text()').getall()
        countUrl = 1
        
        for record in titleToSearch: 
            for title in self.title_list:
                if title.rstrip().strip().lower() == record.rstrip().strip().lower():
                    if result[countUrl][0] == "/":
                        result[countUrl] = "https://vnexpress.net"+result[countUrl]
                        for i in range(1,15):
                            if i == 1:
                                yield scrapy.Request(url=result[countUrl], callback=self.parse)
                            yield scrapy.Request(url=result[countUrl]+"-p"+str(i), callback=self.parse)
                            self.write_file(str(countUrl),self.file_label)
                    else:
                        for i in range(1,15):
                            if i == 1:
                                yield scrapy.Request(url=result[countUrl], callback=self.parse)
                            yield scrapy.Request(url=result[countUrl]+"/p"+str(i), callback=self.parse)
                            self.write_file(str(countUrl),self.file_label)
                    self.write_file(title.rstrip().strip().lower(),self.file_title)         
            countUrl=countUrl+1
                           
    def parse(self, response):
        title_list = response.xpath('//*[@class="title_news"]/a/text()').get()
        self.write_file(support.remove_special_character(title_list),self.file_content)       
 
