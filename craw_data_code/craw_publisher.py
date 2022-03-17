import requests
import re
import json
from lxml import etree
import datetime
import pandas as pd
from lxml import etree
import aiohttp
import asyncio
import time
from urllib.parse import unquote,quote
import difflib
import hashlib
from tqdm import tqdm
import os
from aiohttp import TCPConnector


def get_appid_list():
    app_url = "http://api.steampowered.com/ISteamApps/GetAppList/v2/?key=1C7B85E24EA217F9E3232179A23EFD8C"
    app = requests.get(url=app_url)
    applist = json.loads(app.text)
    appid_list = [x for x in applist['applist']['apps']]
    return appid_list


async def update_prox(session):
    url='http://tiqu.ipidea.io:81/abroad?num=1&type=1&lb=6&sb=0&flow=1&regions=us&port=1'
    for i in range(10):
        try:
            async with session.get(url,timeout=5) as context:
                proxy = await context.text()
            if len(proxy) == 0 or not proxy:
                continue
            proxies = 'http://' + proxy
            print(proxies)
            return proxies
        except Exception as e:
            print(f'get one proxy failed: {e}, times {i+1}')
            return None


async def request_func(url,session,proxies):
    for tries in range(5):
        try:
            if proxies:
                async with session.get(url, proxy=proxies,timeout=10) as context:
                    context_text = await context.text()
            else:
                async with session.get(url, timeout=10) as context:
                    context_text = await context.text()
            return context_text
        except Exception as e:
            if tries < 4:
                continue
            print(f'detail Error: {e}')
            return None


async def get_single_info(url,session,proxies):
    content_text = await request_func(url,session,proxies)
    if content_text:
        tree_info = etree.HTML(content_text)
        name = tree_info.xpath('//*[@class="pageheader curator_name"]//text()')
        try:
            name = name[1]
        except:
            print(url)
            return None
        steam_followers = tree_info.xpath('//*[@class="follow_controls"]')
        if len(steam_followers) > 0:
            steam_followers = int(steam_followers[0].xpath('.//*[@class="num_followers"]/text()')[0].replace(',',''))
        else:
            steam_followers = None
        tweet_followers = tree_info.xpath('//*[contains(@href,"https://twitter.com")]')
        if len(tweet_followers) > 0:
            tweet_followers = tweet_followers[0].xpath('.//*[@class="fan_count"]/text()')
            try:
                tweet_followers = int(tweet_followers[0].replace(',',''))
            except:
                print(tweet_followers)
                tweet_followers = None
        else:
            tweet_followers = None
        return {'name':name,'steam_followers':steam_followers,'tweet_followers':tweet_followers,'url':url}
    else:
        print(url)
        return None


def publisher_links(url):
    content = requests.get(url)
    content = eval(content.text)
    content = content['results_html']
    tree_info = etree.HTML(content)
    urls = tree_info.xpath('//*[@class="highlighted_app_curator_image tooltip"]//@href')
    urls = [x.replace('\\','') for x in urls]
    return urls

async def main(output_file):
    with open(output_file,'w') as f:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
            'Cookie': 'browserid=2578867924147044024; sessionid=ad1deb53d7f012b615c3938e; timezoneOffset=28800,0; _ga=GA1.2.1578446140.1635938587; birthtime=849369601; lastagecheckage=1-0-1997; Steam_Language=english; _gid=GA1.2.861836702.1636175136; steamCountry=US%7C432d617428b7d6b41238ab9a66cdca27; recentapps=%7B%221313%22%3A1636253896%2C%22381210%22%3A1636253390%2C%22458380%22%3A1636253247%2C%223830%22%3A1636252551%2C%222540%22%3A1636252228%2C%22292030%22%3A1636196428%2C%22976310%22%3A1636195708%2C%22985830%22%3A1636195694%2C%221335200%22%3A1636195687%2C%221091500%22%3A1636195678%7D; app_impressions=1252330@1_5_9__300_3|1172470@1_5_9__300_2|924970@1_5_9__300_1|359550@1_5_9__300_5|594650@1_5_9__300_4|397540@1_5_9__300_5|311210@1_5_9__300_4|1172470@1_5_9__300_3|594650@1_5_9__300_2|924970@1_5_9__300_1|359550@1_5_9__300_3|594650@1_5_9__300_2|924970@1_5_9__300_1|686810@1_5_9__300_3|397540@1_5_9__300_2|924970@1_5_9__300_1|397540@1_5_9__300_3|594650@1_5_9__300_2|924970@1_5_9__300_1'
            }
        cookies = {'steamCountry':'US%7C769be7cb2da61e628903aa4f399b8e66',
                   'birthtime':849369601,
                   'lastagecheckage':'1-0-1997',
                   'Steam_Language':'english'}
        proxy = 'http://127.0.0.1:7890'
        tasks = []
        async with aiohttp.ClientSession(headers=headers,cookies=cookies,connector=TCPConnector(verify_ssl=False)) as session:
            for index,i in enumerate(tqdm(range(3000),desc='doing')):
                url = 'https://store.steampowered.com/curators/ajaxgettopcreatorhomes/render/?query=&start='+str(24*i)+'&count=24'
                pub_urls = publisher_links(url)
                print(len(pub_urls))
                if len(pub_urls) == 0:
                    break
                for s_url in pub_urls:
                    task = asyncio.create_task(get_single_info(s_url,session,proxy))
                    tasks.append(task)
                for task in tasks:
                    info = await task
                    if not info:
                        continue
                    else:
                        f.write(json.dumps(info) + '\n')
                tasks = []

if __name__ == '__main__':
    output_file = './publisher.json'
    asyncio.run(main(output_file))