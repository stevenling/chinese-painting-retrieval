"""
从雅昌艺术网爬取国画图片 URL，并将人物类图片下载到本地。

主要流程：
- 先抓取所有作品详情页的 URL
- 再从每个详情页提取图片的真实地址
- 最后按顺序把图片保存到本地目录
"""

import requests
import urllib.request

from bs4 import BeautifulSoup

all_url_list = []
img = []
img_type_list = []
count = 0
first_count = 81  # 人物
second_count = 1  # 山水
third_count = 1  # 花鸟
flag_1 = 0
flag_2 = 0
flag_3 = 0

def download_page(url):
    """
    使用自定义 User-Agent 伪装成浏览器，下载指定页面内容。
    """
    return requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3236.0 Safari/537.36'
    }).content


def get_all_first_url():
    """
    爬取所有作品详情页的 URL，保存到全局列表 all_url_list 中。
    """
    temp_url = "https://gallery.artron.net/works/guohua-all-0-102.html?attributNum=795&attributNam=%E9%A2%98%E6%9D%90&caizhistr=796%2C%E9%A2%98%E6%9D%90%2C"
    # 页数
    page_index = 0
    while page_index < 1:
        url = download_page(temp_url)
        all_soup = BeautifulSoup(url)
        work_list = all_soup.find('div', attrs={'class': 'workList mt20'})
        # print(work_list)
        if work_list is not None:
            all_pic_list = work_list.find_all('div', attrs={'class': 'pic'})
        for all_pic in all_pic_list:
            all_href = all_pic.find('a')  # 一步找到所有的 a 标签
            real_all_href = all_href['href']  # 拿到 url
            if real_all_href not in all_url_list:  # 判断一下是否已经在容器里了，不在的话才加入
                all_url_list.append(real_all_href)  # 存到 all_url_list 这个 list 容器里
        page_index = page_index + 1
        list_jump = all_soup.find('div', attrs={'class': 'listJump'})
        if list_jump is not None:
            next_btn_list = list_jump.find_all('a', attrs={'': ''})
        for next_btn in next_btn_list:
            if next_btn.getText() == "下一页" and next_btn.get("href") is not None:
                temp_url = "https://gallery.artron.net" + next_btn.get("href")


def get_img_url():
    """
    遍历作品详情页 URL，提取每一张国画图片的真实地址，保存到全局列表 img 中。
    """
    for url_index in all_url_list:
        # 首页面格式化
        all_url_digit = download_page(url_index)
        # 得到解析后的 html
        all_soup = BeautifulSoup(all_url_digit)
        all_page = all_soup.find('div', attrs={'class': 'imgCell'})
        if all_page is not None:
            temp_page = all_page.find("img")
            print(temp_page['src'])
            img.append(temp_page['src'])


def main():
    """
    爬取作品页 URL、提取图片地址并下载所有图片。
    """
    get_all_first_url()
    get_img_url()
    download()


def download():
    """
    将全局列表 img 中的图片地址逐个下载到本地，
    当前实现保存为“人物验证X.jpg”文件名，并在控制台输出下载进度。
    """
    global img, count, img_type_list, first_count, second_count, third_count
    print("开始下载图片")
    for image_url in img:
        urllib.request.urlretrieve(image_url, "D:/Desktop//国画图片/" + "人物验证" + str(first_count) + ".jpg")
        first_count = first_count + 1
        # elif (img_type_list[count] == "山水"):
        #     urllib.request.urlretrieve(image_url, "D:/Desktop//国画图片/" + "山水" + str(second_count) + ".jpg")
        #     second_count = second_count + 1
        # elif (img_type_list[count] == "花鸟"):
        #     urllib.request.urlretrieve(image_url, "D:/Desktop//国画图片/" + "花鸟" + str(third_count) + ".jpg")
        #     third_count = third_count + 1
        count = count + 1
        print("正在下载第" + str(count) + "张")
    img = []
    print("下载完毕")


if __name__ == '__main__':
    main()