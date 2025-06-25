 #!/usr/bin/env python      # Mac-Air users may change python to python3
''' 
-----------------------------------
Tutorial Documentation on Comments
-----------------------------------
a. 1 line comments can start with this character: #.
    Anything following the # (pound sign) will be ignored by Python.
b. Multiline comments, such as this section begin and end with
    3 tick marks or single-apostrophies, like this: ...
    and terminate (end) with 3 more tick marks. '''


#-----------------------------------
'''  <--- New comment section start
#-----------------------------------
name:    images_pull_by_content.py

purpose: pull down images from the web
         and save them in a folder with
         the same names as the "keywords" supplied.
            Example: if out "keywords" were 'kittens',
                     then a folder named kittens is 
                     created with the kitten images
                     saved in the "kittens" folder
#-----------------------------------
New comment section ends --> '''
#-----------------------------------

import os


try:
    
    from icrawler.builtin import GoogleImageCrawler
    from icrawler import ImageDownloader

except:
    os.system('pip install icrawler')
    from icrawler.builtin import GoogleImageCrawler
    from icrawler import ImageDownloader


try:
    import urllib.request
    import urllib.parse

except:
    os.system('pip install urllib')
    import urllib.request
    import urllib.parse

try:
    from six.moves.urllib.parse import urlparse

except:
    os.system('pip install six')
    from six.moves.urllib.parse import urlparse

try:
    import base64

except:
    os.system('pip install base64')
    import base64
#==================================#
# Classes
#==================================#
class MyImageDownloader(ImageDownloader):
#-----------------------------------
   search_prefix = 1 #4 but with face, not photo #4 # No License NO Large size # 3 # 'No license and Large' #2 #'License and Large'
   
   def get_filename(self, task, default_ext, search_prefix=search_prefix):
        url_path = urlparse(task['file_url'])[2]
        if '.' in url_path:
            extension = url_path.split('.')[-1]
            if extension.lower() not in [
                    'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'ppm', 'pgm'
            ]:
                extension = default_ext
        else:
            extension = default_ext
        # works for python3
        filename = base64.b64encode(url_path.encode()).decode()
        return '{}.{}'.format('image_' + (f'{search_prefix}') + '_' + (keywords.replace(' ', '-'))+ '_' + filename[:5] + '_' + filename[-5:], extension)
    
###################################
#keywords='ladies portrait glamour shots'
keywords='beyonce portrait glamour shots'
###################################

###################################
# M A I N   L O G I C   H E R E
###################################
google_crawler = GoogleImageCrawler(
    downloader_cls=MyImageDownloader,
    feeder_threads=2, #1,
    parser_threads=4, #2,
    downloader_threads=8, #4,
    storage={'root_dir': './images' + '/' + keywords})
filters = dict(
    type='face',
    #type='photo',
    #type='linedrawing',
    #size='large',
    size='medium', #icon', # 'large', 'medium',
    #color='red',
    #license='commercial,modify',
    #-----------------------------------
    date=((2022, 1, 1), (2025, 6, 24)))  # Hey! I do not want last decade's styles. Keep it current
    #-----------------------------------
google_crawler.crawl(keyword=keywords, filters=filters, max_num=200, file_idx_offset=10)
