# https://gallery.artron.net/works/guohua.html?attributNum=795&attributNam=%E9%A2%98%E6%9D%90&caizhistr=797%2C%E9%A2%98%E6%9D%90%2C
# !/usr/bin/env python
# encoding=utf-8
# python 爬取 https://shop.artron.net/works/10088_w761428.html#nextworklink

import requests
import urllib.request

from bs4 import BeautifulSoup

all_url_list = []
img = []
img_type_list = []
count = 0
first_count = 1  # 人物
second_count = 1  # 山水
third_count = 1  # 花鸟
flag_1 = 0
flag_2 = 0
flag_3 = 0


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
        if work_list is not None:
            all_pic = work_list.find_all('div', attrs={'class': 'pic'})
        for all_pic_index in all_pic:
            # 一步找到所有的a标签
            all_href = all_pic_index.find('a')
            # 拿到作品详情页的 url
            real_all_href = all_href['href']
            # 判断一下是否已经在容器里了，不在的话才加入
            if real_all_href not in all_url_list:  
                # 存到 all_url_list 这个 list 容器里
                all_url_list.append(real_all_href)  
        page_index = page_index + 1
        list_jump = all_soup.find('div', attrs={'class': 'listJump'})
        if list_jump is not None:
            next_btn_list = list_jump.find_all('a', attrs={'': ''})
        for next_btn in next_btn_list:
            if next_btn.getText() == "下一页" and next_btn.get("href") is not None:
                temp_url = "https://gallery.artron.net" + next_btn.get("href")


def get_img_url():
    """
    获取每一张图片的 url
    """
    for url_index in all_url_list:
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
    global img, count, img_type_list, first_count, second_count, third_count
    print("开始下载图片")
    for current_img_file in img:
        urllib.request.urlretrieve(current_img_file, "D:/Desktop//国画图片/" + "花鸟" + str(first_count) + ".jpg")
        first_count = first_count + 1
        # elif (img_type_list[count] == "山水"):
        #     urllib.request.urlretrieve(m, "D:/Desktop//国画图片/" + "山水" + str(second_count) + ".jpg")
        #     second_count = second_count + 1
        # elif (img_type_list[count] == "花鸟"):
        #     urllib.request.urlretrieve(m, "D:/Desktop//国画图片/" + "花鸟" + str(third_count) + ".jpg")
        #     third_count = third_count + 1
        count = count + 1
        print("正在下载第" + str(count) + "张")
    img = []
    print("下载完毕")


if __name__ == '__main__':
    main()
    # download();
