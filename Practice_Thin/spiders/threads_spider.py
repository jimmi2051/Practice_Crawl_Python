import scrapy
import os


class ThreadsSpider(scrapy.Spider):
    name = "threads"
    folder_path = "input"
    folder_result = "output"
    title_list = []
    page_default = "https://vnexpress.net/"
    url_list = [] 
    def start_requests(self):
        
        input_title = open("input/title.txt", 'r+')
        data_title =input_title.readlines()
        for line in data_title:
            self.title_list.append(line)
        yield scrapy.Request(url=self.page_default,callback=self.getUrl)

    def getUrl(self,response):
        result = response.xpath('//*[@id="main_menu"]/a/@href').getall()
        titleToSearch = response.xpath('//*[@id="main_menu"]/a/text()').getall()
        countUrl = 0
        
        for record in titleToSearch: 
            for title in self.title_list:
                if title.rstrip().strip().lower() == record.rstrip().strip().lower():
                    if result[countUrl][0] == "/":
                        result[countUrl] = "https://vnexpress.net"+result[countUrl]
                    self.url_list.append(result[countUrl])
            countUrl=countUrl+1
        for url in self.url_list:
            for i in range(1,5):
                if i == 1:
                    yield scrapy.Request(url=url, callback=self.parse)
                yield scrapy.Request(url=url+"-p"+str(i), callback=self.parse)
                           
        self.log('Saved file %s' % filename)
    def parse(self, response):

        page = response.url.split("/")

        filename = page[len(page)-1] + ".html"

        with open(self.folder_result+"/"+filename, 'wb') as f:
            f.write(response.body)

        self.log('Saved file %s' % filename)
