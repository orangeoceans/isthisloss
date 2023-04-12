from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_url = "https://knowyourmeme.com"
loss_url = "/memes/loss/photos/page/"
kym_cdn_url = "https://i.kym-cdn.com/photos/images/"
allowed_suffixes = [".png", ".jpg", ".jpeg"]
headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0'
    }

output_dir = "./loss_imgs"


def check_img_url(img_url: int) -> bool:
    """
    Make sure only kym-cdn urls and appropriate file types are accepted.
    No GIFs for now.
    """
    if not img_url.startswith(kym_cdn_url):
        return False
    if not any([img_url.endswith(suffix) for suffix in allowed_suffixes]):
        return False
    return True

def main():
    sess = requests.session()
    
    img_count = 0
    for page_num in range(1,50):
        page_img_count = 0
        try:
            response = sess.get(f"{base_url}{loss_url}{page_num}", headers=headers, timeout=1)
        except RequestException as exception:
            logger.warning(exception)
            continue
        img_pages_soup = BeautifulSoup(response.text, "lxml")
        infinite_scroll = img_pages_soup.find(id="infinite-scroll-wrapper")
        for tag in infinite_scroll.findAll("div", {"class":"item"}):
            page_link = tag.find("a")["href"]
            try:
                response = sess.get(f"{base_url}{page_link}", headers=headers, timeout=1)
                img_url_soup = BeautifulSoup(response.text, "lxml")
                img_url = img_url_soup.find("img", {"class":"centered_photo"})["src"]
                
                if check_img_url(img_url):
                    response = sess.get(img_url, headers=headers, timeout=1)
                    output_path = f"{output_dir}/{img_count}.jpg"
                    with open(output_path, "wb") as file:
                        file.write(response.content)
                    page_img_count += 1
                    img_count += 1
                    
            except RequestException as exception:
                logger.warning(exception)
                continue
                
        logger.info(f"Downloaded {page_img_count} images on page {page_num}.")
        
            

if __name__ == "__main__":
    main()