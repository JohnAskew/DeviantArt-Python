import os

try:
    
    os.system('pip install icrawler')
    
    from icrawler.builtin import GoogleImageCrawler

except:

    os.system('pip install icrawler')

    from icrawler.builtin import GoogleImageCrawler


google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': './images'})
filters = dict(
    size='large',
    color='red',
    license='commercial,modify',
    date=((2022, 1, 1), (2022, 5, 20)))
google_crawler.crawl(keyword='julia set', filters=filters, max_num=100, file_idx_offset=0)