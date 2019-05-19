# https://gallery.artron.net/works/guohua.html?attributNum=795&attributNam=%E9%A2%98%E6%9D%90&caizhistr=797%2C%E9%A2%98%E6%9D%90%2C
# !/usr/bin/env python


# encoding=utf-8



# python爬取 https://shop.artron.net/works/10088_w761428.html#nextworklink

import requests
from bs4 import BeautifulSoup
import urllib.request

allUrl = []
img = []
imgType=[]
count= 0
firstCount = 1  #人物
secondCount = 1 #山水
thirdCount = 1 #花鸟
flag1 = 0
flag2 = 0
flag3 = 0
#伪装成浏览器

#人物
#https://gallery.artron.net/works/guohua.html?attributNum=795&attributNam=%E9%A2%98%E6%9D%90&caizhistr=796%2C%E9%A2%98%E6%9D%90%2C


def download_page(url):
    return requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3236.0 Safari/537.36'
    }).content



# 爬取所有图集的url，放到一个allUrl这个list里

def get_all_first_url():
    tempUrl = "https://gallery.artron.net/works/guohua.html?attributNum=797&attributNam=%E9%A2%98%E6%9D%90&caizhistr=796%2C%E9%A2%98%E6%9D%90%2C"
    pageindex = 0 #页数
    while pageindex < 1:
        url = download_page(tempUrl)
        allSoup = BeautifulSoup(url)
        workList = allSoup.find('div', attrs={'class': 'workList mt20'})
        #print(workList)
        if workList != None:
            allPic = workList.find_all('div', attrs = {'class':'pic'})
        for allPicIndex in allPic:
            allHref = allPicIndex.find('a')  # 一步找到所有的a标签
            realAllHref = allHref['href']  # 拿到url
            if realAllHref not in allUrl:  # 判断一下是否已经在容器里了，不在的话才加入
                allUrl.append(realAllHref)  # 存到allurl这个list容器里
        pageindex = pageindex + 1
        listJump = allSoup.find('div', attrs={'class': 'listJump'})
        #print(listJump)
        if listJump != None:
            nextBtn = listJump.find_all('a',attrs={'':''})
        for nextBtnIndex in nextBtn:
        #print(nextBtn)
            if nextBtnIndex.getText() == "下一页" and nextBtnIndex.get("href") != None:
               tempUrl = "https://gallery.artron.net" + nextBtnIndex.get("href")
        #print(tempUrl)

#获得每一张图片url
def get_img_url():
    for urlIndex in allUrl:
        allurldigit = download_page(urlIndex)  # 首页面格式化
        allsoup = BeautifulSoup(allurldigit)  # 得到解析后的html
        allpage = allsoup.find('div', attrs={'class': 'imgCell'})
        if allpage != None:
            tempPage = allpage.find("img")
            print(tempPage['src'])
            img.append(tempPage['src'])
    #print(allUrl)


def main():
    get_all_first_url();
    get_img_url()
    download()
def download():
    #print(len(img))
    global img,count,imgType,firstCount,secondCount,thirdCount
    print("开始下载图片")
    for m in img:
        urllib.request.urlretrieve(m, "D:/Desktop//国画图片/" + "花鸟"+ str(firstCount) + ".jpg")
        firstCount = firstCount + 1
        # elif (imgType[count] == "山水"):
        #     urllib.request.urlretrieve(m, "D:/Desktop//国画图片/" + "山水" + str(secondCount) + ".jpg")
        #     secondCount = secondCount + 1
        # elif (imgType[count] == "花鸟"):
        #     urllib.request.urlretrieve(m, "D:/Desktop//国画图片/" + "花鸟" + str(thirdCount) + ".jpg")
        #     thirdCount = thirdCount + 1
        count = count+1
        print("正在下载第"+str(count)+"张")
    img = []
    print("下载完毕")

if __name__ == '__main__':
    main()
    #download();
