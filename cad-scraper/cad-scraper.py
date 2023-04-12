from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_url = "https://cad-comic.com/comic/all-about-the-handle/"
cad_img_url = "https://cad-comic.com/wp-content/uploads/"
allowed_suffixes = [".png", ".jpg", ".jpeg"]
headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0'
    }

output_dir = "./cad_imgs"
log_progress_step = 50
num_outputs = 1000


def check_img_url(img_url: int) -> bool:
    """
    Make sure only kym-cdn urls and appropriate file types are accepted.
    No GIFs for now.
    """
    if not img_url.startswith(cad_img_url):
        return False
    if not any([img_url.endswith(suffix) for suffix in allowed_suffixes]):
        return False
    return True

def main():
    """
    Starts at start_url and scrapes every successive comic up to num_outputs.

    """
    sess = requests.session()
    
    this_url = start_url
    for img_num in range(num_outputs):
        try:
            response = sess.get(this_url, headers=headers, timeout=1)
        except RequestException as exception:
            logger.warning(f"Scraping aborted due to exception: {exception}")
            break
        comic_page_soup = BeautifulSoup(response.text, "lxml")
        
        comic_page = comic_page_soup.find("div", {"class":"comicpage"})
        comic_a_tag = comic_page.find("a",{"href":this_url})
        img_url = comic_a_tag.find("img")["src"]    
        if check_img_url(img_url):
            response = sess.get(img_url, headers=headers, timeout=1)
            output_path = f"{output_dir}/{img_num}.jpg"
            with open(output_path, "wb") as file:
                file.write(response.content)
        
        this_url = comic_page.find("a", {"rel":"next"})["href"]
        if img_num%log_progress_step == 0:
            print(f"Downloaded {img_num+1} comic(s) thus far.")
            

if __name__ == "__main__":
    main()