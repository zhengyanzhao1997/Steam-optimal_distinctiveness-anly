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
from content_func import get_onepackage,get_onegame
import os


def get_appid_list():
    app_url = "http://api.steampowered.com/ISteamApps/GetAppList/v2/?key=1C7B85E24EA217F9E3232179A23EFD8C"
    app = requests.get(url=app_url)
    applist = json.loads(app.text)
    appid_list = [x for x in applist['applist']['apps']]
    return appid_list


async def request_func(appid,session,proxies):
    url = 'https://store.steampowered.com/app/' + str(appid)
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


async def get_single_info(app_id,name,session,proxies):
    pre_difined_item = {'game_id': app_id, 'name': name}
    content_text = await request_func(app_id,session,proxies)
    if content_text:
        tree_info = etree.HTML(content_text)
        name = tree_info.xpath('//*[@id="appHubAppName"]/text()')
        if len(name) == 0:
            pakeage_header = tree_info.xpath('//*[@id="package_header_container"]')
            ops = tree_info.xpath('//*[@id="error_box"]')
            if 'Welcome to Steam' in content_text:
                pre_difined_item['request_status'] = 'Not yet on steam'
            elif (len(ops) > 0 and ops[0] == 'Oops, sorry!') or 'Oops, sorry!' in content_text:
                pre_difined_item['request_status'] = 'Not in your region'
            elif len(pakeage_header) > 0:
                #print('package',app_id)
                pre_difined_item['request_status'] = 'Package'
                pre_difined_item = get_onepackage(content_text,pre_difined_item)
            else:
                pre_difined_item['request_status'] = 'Other'
                #print('Other',app_id)
        else:
            pre_difined_item['request_status'] = 'Available'
            pre_difined_item = get_onegame(content_text,pre_difined_item)
    else:
        return None
    return pre_difined_item


async def main(output_file,appid_list):
    app_size = len(appid_list)
    print('total num',app_size)
    app_got_list = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for i in f:
                app_got_list.append(json.loads(i)['game_id'])
    print('already exit num',len(app_got_list))
    with open(output_file,'a+') as f:
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
        async with aiohttp.ClientSession(headers=headers,cookies=cookies) as session:
            for index,id_ in enumerate(tqdm(appid_list,desc='doing')):
                if id_['appid'] in app_got_list:
                    continue
                task = asyncio.create_task(get_single_info(id_['appid'], id_['name'],session,proxy))
                tasks.append(task)
                if len(tasks) == 200 or index >= app_size - 1:
                    for task in tasks:
                        info = await task
                        if not info:
                            continue
                        else:
                            f.write(json.dumps(info) + '\n')
                    tasks = []
            for task in tasks:
                info = await task
                if not info:
                    continue
                else:
                    f.write(json.dumps(info) + '\n')
