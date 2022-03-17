import requests
import json
from lxml import etree
from content_func import get_onepackage,get_onegame
import aiohttp
import asyncio
import re
from tqdm import tqdm

async def get_app_list(session,nurl):
    content = await request_func(session, nurl)
    app_list = re.findall(r'data-ds-appid=\\"(.*?)\\"',content)
    return app_list

async def request_func(session,url):
    for tries in range(5):
        try:
            async with session.get(url,proxy='http://127.0.0.1:7890',timeout=10) as context:
                context_text = await context.text()
            return context_text
        except Exception as e:
            if tries < 4:
                continue
            print(f'detail Error: {e}')
            return None

async def use_yibu():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
        'Cookie': 'browserid=2578867924147044024; sessionid=ad1deb53d7f012b615c3938e; timezoneOffset=28800,0; _ga=GA1.2.1578446140.1635938587; birthtime=849369601; lastagecheckage=1-0-1997; Steam_Language=english; _gid=GA1.2.861836702.1636175136; steamCountry=US%7C432d617428b7d6b41238ab9a66cdca27; recentapps=%7B%221313%22%3A1636253896%2C%22381210%22%3A1636253390%2C%22458380%22%3A1636253247%2C%223830%22%3A1636252551%2C%222540%22%3A1636252228%2C%22292030%22%3A1636196428%2C%22976310%22%3A1636195708%2C%22985830%22%3A1636195694%2C%221335200%22%3A1636195687%2C%221091500%22%3A1636195678%7D; app_impressions=1252330@1_5_9__300_3|1172470@1_5_9__300_2|924970@1_5_9__300_1|359550@1_5_9__300_5|594650@1_5_9__300_4|397540@1_5_9__300_5|311210@1_5_9__300_4|1172470@1_5_9__300_3|594650@1_5_9__300_2|924970@1_5_9__300_1|359550@1_5_9__300_3|594650@1_5_9__300_2|924970@1_5_9__300_1|686810@1_5_9__300_3|397540@1_5_9__300_2|924970@1_5_9__300_1|397540@1_5_9__300_3|594650@1_5_9__300_2|924970@1_5_9__300_1'
        }
    cookies = {'steamCountry':'US%7C769be7cb2da61e628903aa4f399b8e66',
               'birthtime':849369601,
               'lastagecheckage':'1-0-1997',
               'Steam_Language':'english'}

    url = 'https://store.steampowered.com/'
    async with aiohttp.ClientSession(headers=headers,cookies=cookies) as session:
        async with session.get(url,proxy='http://127.0.0.1:7890',timeout=10) as context:
            context_text = await context.text()
            tree_info = etree.HTML(context_text)
            subcat = tree_info.xpath('//*[@class="popup_genre_expand_content responsive_hidden"]//a//@href')
            url_base = []
            for surl in subcat:
                nurl = surl.split('/')[-2]
                url_base.append(nurl)


    cat_dict = {}
    async with aiohttp.ClientSession(headers=headers,cookies=cookies) as session:
        for base_url in tqdm(url_base):
            app_lists = []
            furl = 'https://store.steampowered.com/category/' + base_url + "/#p=0&tab=TopSellers"
            context_text = await request_func(session,furl)
            tree_info = etree.HTML(context_text)
            pageitem = tree_info.xpath('//*[@id="TopSellers_total"]/text()')[0]
            max_num = int(pageitem.replace(',',''))
            pageitem = tree_info.xpath('//*[@id="TopSellers_end"]/text()')[0]
            evry_page = int(pageitem.replace(',',''))
            max_page = int(max_num/evry_page) + 1
            tasks = []
            info = await get_app_list(session,'https://store.steampowered.com/contenthub/querypaginated/category/TopSellers/render/?query=&start=840&count=15&cc=US&l=english&v=4&tag=Action%20Roguelike&tagid=42804')
            print(info)
            for page in range(max_page):
                start = 15 * page
                nurl = "https://store.steampowered.com/contenthub/querypaginated/category/TopSellers/render/?query=&start=" + str(start) + "&count=15&cc=US&l=english&v=4&tag=&category=" + base_url
                task = asyncio.create_task(get_app_list(session,nurl))
                tasks.append(task)
                if len(tasks) == 50 or page >= max_page - 1:
                    for task in tasks:
                        info = await task
                        if not info:
                            continue
                        else:
                            app_lists += info
                    tasks = []
            print(base_url)
            print(len(app_lists))
            print(max_num)
            #assert len(app_lists) == max_num
            cat_dict[base_url] = app_lists

    with open('cat_dict_new2.json','w') as f:
        f.write(json.dumps(cat_dict))
asyncio.run(use_yibu())