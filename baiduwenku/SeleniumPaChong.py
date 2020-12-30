from selenium import webdriver
from time import sleep
from bs4 import BeautifulSoup

urlList = []

def getHtmlBySelenium(url):
    browser = webdriver.Chrome(executable_path="D:\\work\\chromePlunge\\chromedriver_win32\\chromedriver.exe")
    browser.get(url)
    soup = BeautifulSoup(browser.page_source,"html.parser")
    #print(soup)
    plist = []
    #标题
    for div in soup.find_all('div', attrs={"class": "doc-title-wrap"}):
        plist.extend(div.get_text().split('\n'))
    #文档信息
    for div in soup.find_all('div', attrs={"class": "doc-summary-wrap"}):
        plist.extend(div.get_text().split('\n'))
    #价格
    div_price = soup.find_all('div', attrs={"class": "doc-btns-wrap"})
    for div in div_price:
        plist.extend(div.get_text().split('\n'))

    plist = [c.replace(' ', '') for c in plist]
    plist = [c.replace('\x0c', '') for c in plist]
    plist = [i for i in plist if i != ''];

    browser.close();
    #print(plist)
    return plist

def printPList(plist, path = 'baiduwenku1.txt'):
    file = open(path, 'a')
    file.write('\n')
    for str in plist:
        file.write(str)
        file.write('\t')
    file.close()

def getPageUrl(url):
    browser = webdriver.Chrome(executable_path="D:\\work\\chromePlunge\\chromedriver_win32\\chromedriver.exe")
    browser.get(url)
    elems=browser.find_elements_by_class_name('img-title')
    main_windows = browser.current_window_handle
    for e in elems:
        e.click()

    all_handles = browser.window_handles
    for handle in all_handles:
        if handle != main_windows:
            browser.switch_to.window(handle)
            urlList.append(browser.current_url)
    return ''

def main():
    for i in urlList:
        #url = 'https://wenku.baidu.com/view/2a8a773c4973f242336c1eb91a37f111f0850d5a'
        plist = getHtmlBySelenium(i)
        printPList(plist)

if __name__ == '__main__':
    getPageUrl('https://wenku.baidu.com/ndcore/browse/sub?isBus=1&isPus=3&isChild=&isType=2')
    #print(urlList)
    main()