import requests
import json
from lxml import etree
from content_func import get_onepackage,get_onegame
import aiohttp
import asyncio
from tqdm import tqdm

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

def clean_trn(str_: str) -> str:
    str_ = str_.replace('\t', '').replace('\r', '').replace('\n', '').strip()
    return str_

def get_applist(context_text):
    item_lists = []
    tree_info = etree.HTML(context_text)
    game_titles = tree_info.xpath('//*[@class="clamp-summary-wrap"]')
    for game in game_titles:
        meta_score = game.xpath('.//*[contains(@class,"metascore_w")]//text()')[0]
        title_url = game.xpath('.//*[@class="title"]/@href')[0]
        game_name = title_url.split("/")[-1]
        summary = clean_trn(game.xpath('.//*[@class="summary"]//text()')[0])
        item = {'game_name': game_name, 'title_url': title_url,'meta_score':meta_score,'summary':summary}
        item_lists.append(item)
    return item_lists

async def get_single_info(url, session, proxies):
    content_text = await request_func(url, session, proxies)
    item = get_applist(content_text)
    return item

async def use_yibu():
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',}
    pages = 68
    proxy = None
    tasks = []
    result = []
    async with aiohttp.ClientSession(headers=headers) as session:
        for index, page in enumerate(tqdm(range(pages), desc='doing')):
            url = 'https://www.metacritic.com/browse/games/release-date/available/pc/metascore?page=' + str(page)
            task = asyncio.create_task(get_single_info(url, session, proxy))
            tasks.append(task)
            if len(tasks) == 20 or index == 67:
                for task in tasks:
                    info = await task
                    if not info:
                        continue
                    else:
                        result.extend(info)
                tasks = []
    with open('./game_title_lists.json','w') as f:
        for item in result:
            f.write(json.dumps(item) + '\n')
asyncio.run(use_yibu())
