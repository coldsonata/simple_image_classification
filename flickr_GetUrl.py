## > python flickr_GetUrl.py tag number_of_images_to_attempt_to_download
from flickrapi import FlickrAPI
import pandas as pd
import sys
key='60e4a2c2c30f6983e4e5d92200a708a9'
secret='4465144dc26ff883'
def get_urls(image_tag,MAX_COUNT):
    flickr = FlickrAPI(key, secret)
    photos = flickr.walk(text=image_tag,
                            tag_mode='all',
                            tags=image_tag,
                            extras='url_o, url_l, url_m, url_s',
                            per_page=50,
                            sort='relevance')
    count=0
    urls=[]
    for photo in photos:
        if count< MAX_COUNT:
            count=count+1
            print("Fetching url for image number {}".format(count))
            try:
                url=photo.get('url_o') or photo.get('url_l') or photo.get('url_m') or photo.get('url_s')
                urls.append(url)
            except:
                print("Url for image number {} could not be fetched".format(count))
        else:
            print("Done fetching urls, fetched {} urls out of {}".format(len(urls),MAX_COUNT))
            break
    urls=pd.Series(urls)
    print("Writing out the urls in the current directory")
    urls.to_csv(image_tag+"_urls.csv")
    print("Done!!!")
def main():
    tag=sys.argv[1]
    MAX_COUNT=int(sys.argv[2])
    get_urls(tag,MAX_COUNT)
if __name__=='__main__':
    main()