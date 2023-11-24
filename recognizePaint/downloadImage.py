# https://gallery.artron.net/works/guohua.html?attributNum=795&attributNam=%E9%A2%98%E6%9D%90&caizhistr=797%2C%E9%A2%98%E6%9D%90%2C
# !/usr/bin/env python
# encoding=utf-8
# python 爬取 https://shop.artron.net/works/10088_w761428.html#nextworklink

import requests
import urllib.request

from bs4 import BeautifulSoup

allUrl = []
img = []
imgType = []
count = 0
firstCount = 1  # 人物
secondCount = 1  # 山水
thirdCount = 1  # 花鸟
flag1 = 0
flag2 = 0
flag3 = 0


# 伪装成浏览器

# 人物
# https://gallery.artron.net/works/guohua.html?attributNum=795&attributNam=%E9%A2%98%E6%9D%90&caizhistr=796%2C%E9%A2%98%E6%9D%90%2C

def download_page(url):
    return requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3236.0 Safari/537.36'
    }).content


# 爬取所有图集的url，放到一个allUrl这个list里
def get_all_first_url():
    temp_url = "https://gallery.artron.net/works/guohua.html?attributNum=797&attributNam=%E9%A2%98%E6%9D%90&caizhistr=796%2C%E9%A2%98%E6%9D%90%2C"
    # 页数
    page_index = 0
    while page_index < 1:
        url = download_page(temp_url)
        all_soup = BeautifulSoup(url)
        work_list = all_soup.find('div', attrs={'class': 'workList mt20'})
        # print(workList)
        if work_list is not None:
            all_pic = work_list.find_all('div', attrs={'class': 'pic'})
        for all_pic_index in all_pic:
            # 一步找到所有的a标签
            all_href = all_pic_index.find('a')
            realAllHref = all_href['href']  # 拿到url
            if realAllHref not in allUrl:  # 判断一下是否已经在容器里了，不在的话才加入
                allUrl.append(realAllHref)  # 存到allurl这个list容器里
        page_index = page_index + 1
        listJump = all_soup.find('div', attrs={'class': 'listJump'})
        # print(listJump)
        if listJump is not None:
            nextBtn = listJump.find_all('a', attrs={'': ''})
        for nextBtnIndex in nextBtn:
            # print(nextBtn)
            if nextBtnIndex.getText() == "下一页" and nextBtnIndex.get("href") is not None:
                temp_url = "https://gallery.artron.net" + nextBtnIndex.get("href")
        # print(tempUrl)


def get_img_url():
    """
    获取每一张图片的 url
    """
    for url_index in allUrl:
        # 首页面格式化
        all_url_digit = download_page(url_index)
        # 得到解析后的 html
        all_soup = BeautifulSoup(all_url_digit)
        all_page = all_soup.find('div', attrs={'class': 'imgCell'})
        if all_page is not None:
            temp_page = all_page.find("img")
            if temp_page is not None:
                print(temp_page['src'])
                img.append(temp_page['src'])


def main():
    get_all_first_url();
    get_img_url()
    download()


def download():
    # print(len(img))
    global img, count, imgType, firstCount, secondCount, thirdCount
    print("开始下载图片")
    for current_img_file in img:
        urllib.request.urlretrieve(current_img_file, "D:/Desktop//国画图片/" + "花鸟" + str(firstCount) + ".jpg")
        firstCount = firstCount + 1
        # elif (imgType[count] == "山水"):
        #     urllib.request.urlretrieve(m, "D:/Desktop//国画图片/" + "山水" + str(secondCount) + ".jpg")
        #     secondCount = secondCount + 1
        # elif (imgType[count] == "花鸟"):
        #     urllib.request.urlretrieve(m, "D:/Desktop//国画图片/" + "花鸟" + str(thirdCount) + ".jpg")
        #     thirdCount = thirdCount + 1
        count = count + 1
        print("正在下载第" + str(count) + "张")
    img = []
    print("下载完毕")


if __name__ == '__main__':
    main()
    # download();
