import requests
from bs4 import BeautifulSoup
import bs4
from requests_html import HTMLSession

def getRenderHtml(url):
    session = HTMLSession()
    first_page = session.get('https://sou.zhaopin.com/?jl=763&kw=%E7%88%AC%E8%99%AB%E5%B7%A5%E7%A8%8B%E5%B8%88&kt=3')
    first_page.html.render(sleep=5)
    return first_page.html.html

def getHTMLText(url):
    kv = {'User-agent': 'Baiduspider'}
    try:
        r = requests.get(url, headers = kv, timeout = 30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ''

def findPList(html):
    plist = []
    soup = BeautifulSoup(html, "html.parser")
    plist.append(soup.title.string)
    for div in soup.find_all('div', attrs={"class": "bd doc-reader"}):
        plist.extend(div.get_text().split('\n'))

    plist = [c.replace(' ', '') for c in plist]
    plist = [c.replace('\x0c', '') for c in plist]
    return plist

def printPList(plist, path = 'baiduwenku.txt'):
    file = open(path, 'w')
    for str in plist:
        file.write(str)
        file.write('\n')
    file.close()

def main():

    url = 'https://wenku.baidu.com/ndcore/browse/sub?isBus=1&isPus=3&isChild=&isType=2'
    #html = getHTMLText(url)
    html = getRenderHtml(url)
    plist = findPList(html)
    printPList(plist)

if __name__ == '__main__':
    main()