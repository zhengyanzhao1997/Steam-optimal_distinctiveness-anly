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

def get_applist(game,context_text):
    tree_info = etree.HTML(context_text)
    if tree_info:
        user_reviews = tree_info.xpath('//*[contains(@class,"review user_review")]')
    else:
        return None
    reviews = []
    for review in user_reviews:
        review_grade = int(review.xpath('.//*[@class="review_grade"]//*[contains(@class,"metascore_w")]//text()')[0])
        review_body = review.xpath('.//*[@class="review_body"]')[0]
        blurb_expanded = review_body.xpath('.//*[@class="blurb blurb_expanded"]//text()')
        if len(blurb_expanded) > 0:
            review_text = blurb_expanded
        else:
            review_text = review_body.xpath('.//text()')
        review_text = ' '.join([clean_trn(x) for x in review_text if clean_trn(x).replace(' ', '').replace(',', '') != ""])
        total_ups = int(review.xpath('.//*[@class="total_ups"]//text()')[0])
        total_thumbs = int(review.xpath('.//*[@class="total_thumbs"]//text()')[0])
        item = {'review_grade': review_grade, 'review_text': review_text,'total_ups':total_ups,'total_thumbs':total_thumbs,'game_info':game}
        reviews.append(item)
    return reviews

async def get_single_info(game, url, session, proxies):
    content_text = await request_func(url, session, proxies)
    item = get_applist(game,content_text)
    return item

game_lists = []
with open('./game_title_lists.json','r') as f:
    for i in f:
        game_lists.append(json.loads(i))

async def use_yibu():
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',}
    proxy = 'http://127.0.0.1:7890'
    #proxy = None
    tasks = []
    result = []
    async with aiohttp.ClientSession(headers=headers) as session:
        for index, game in enumerate(tqdm(game_lists, desc='doing')):
            first_url = "https://www.metacritic.com/game/pc/" + game["game_name"] + "/user-reviews?page=0"
            first_text = await request_func(first_url, session, proxy)
            tree_info = etree.HTML(first_text)
            if not tree_info:
                continue
            max_page = tree_info.xpath('//*[@class="page last_page"]//text()')
            if len(max_page) > 0:
                max_page = int(max_page[-1])
            else:
                max_page = 1
            for page in range(max_page):
                url = "https://www.metacritic.com/game/pc/" + game["game_name"] + "/user-reviews?page=" + str(page)
                task = asyncio.create_task(get_single_info(game, url, session, proxy))
                tasks.append(task)
                if len(tasks) == 50:
                    for task in tasks:
                        info = await task
                        if not info:
                            continue
                        else:
                            result.extend(info)
                    tasks = []
            print(len(result))
    with open('./comment_result.json','w') as f:
        for item in result:
            f.write(json.dumps(item) + '\n')
asyncio.run(use_yibu())
