from bs4 import BeautifulSoup
import requests
import time

#Google are assholes, so we can get about ~600 covers before they ban us
def get_img_url_from_google(name):
    query = "q={} movie poster&tbm=isch".format(name)
    time.sleep(1)
    print("https://www.google.ie/search?" + query)
    page = requests.get("https://www.google.ie/search?" + query)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup.find_all(class_="images_table")[0].find_all("img")[0]["src"]

def download_file(url, local_filename):
    r = requests.get(url)
    print("Saving: ", local_filename)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return

titles = []
scores = []

for nr in range(0,300):
    print("On page number: ", nr)
    page = requests.get("https://www.rottentomatoes.com/browse/box-office/?rank_id={}&country=us".format(nr))
    soup = BeautifulSoup(page.content, 'html.parser')

    tile_elements = soup.find_all(target="_top")
    list_items = soup.find_all(itemprop="itemListElement") #Does not contain the <a> elements... because weird
    substr = 0

    for i in range(0,len(tile_elements)):
        if(len(list_items) - 1 < (i - substr)):
            break
        score_element_list = list_items[i - substr].findChildren(class_="tMeterScore") #Either 1 or empty
        if(len(score_element_list) == 0):
            substr += 1
            continue

        fmt_title = tile_elements[i].text
        if fmt_title in titles:
            print("Found title repeat: ", fmt_title)
            continue

        titles.append(fmt_title)
        scores.append(str(score_element_list[0].text[:-1]))

print(len(titles), len(scores))
with open('data.txt', 'a') as f:
    f.write('{"movies":[')
    objs = []
    for i in range(0,len(titles)):
        try:
            title = titles[i]
            print("Saving image for title: ", title)
            score = scores[i]
            print("With scroe: ", score)
            url = get_img_url_from_google(title)
            download_file(url, "original/{}@{}.jpg".format(title, score))
            objs.append('{{"title":"{}", "score":{}, "url":"{}" }}'.format(title, score, url))
        except Exception as e:
            print("There was an error: {} \nGoing to skip this movie".format(e))
    f.write(','.join(objs) + "]}")
