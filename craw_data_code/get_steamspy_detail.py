# Objective: download every Steam review for the games of your choice.
#
# Input:
#   - idlist.txt
#
# Output:
#   - idprocessed.txt
#   - data/review_APPID.json for each APPID in idlist.txt
#
# Reference:
#   https://raw.githubusercontent.com/CraigKelly/steam-data/master/data/games.py

import json
import aiohttp
import asyncio
import requests
import os
from tqdm import tqdm
from steamspypi.compatibility import check_request, fix_request

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}

class Proxy:
    def __init__(self):
        self.proxy_pool = self.get_one_proxy()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.proxy_pool) == 0:
            self.update_pool()
        return self.proxy_pool.pop()

    def update_pool(self):
        self.proxy_pool = self.get_one_proxy()

    def get_one_proxy(self):
        url = 'http://tiqu.ipidea.io:81/abroad?num=300&type=1&lb=4&sb=0&flow=1&regions=us&port=1'
        for i in range(10):
            try:
                rsp = requests.get(url=url, timeout=5)
                proxy = rsp.text
                proxy = proxy.split('\n')
                if len(proxy) < 2:
                    continue
                proxy = ['http://' + x for x in proxy]
                return proxy
            except Exception as e:
                print('get one proxy failed: {}, times {}'.format(e,i+1))
                raise

proxy_pool = Proxy()

def get_data(input_data):
    id_lists = []
    with open(input_data, 'r') as f:
        for i in f:
            id_lists.append(int(i.replace('\n','')))
    return id_lists


async def request_func(session,url,params):
    for tries in range(10):
        try:
            proxy = next(proxy_pool)
            async with session.get(url,proxy=proxy,timeout=10,params=params) as context:
                output = await context.json()
                if context.status != 200:
                    continue
                return context.status,output
        except Exception as e:
            if tries < 9:
                continue
            print('detail Error: {}'.format(e))
            return None,None


async def download_for_app_id(session,url,app_id,params):
    status,content = await request_func(session,url,params)
    if content is not None:
        data = content
        data['query_id'] = app_id
        return data
    else:
        print('cant get', params['appid'])
        return None

def get_api_url():
    api_url = "https://steamspy.com"
    return api_url


def get_api_endpoint():
    api_endpoint = "/api.php"
    return api_endpoint


async def demo(app_lists,output_file):
    tasks = []
    app_size = len(app_lists)
    url = get_api_url() + get_api_endpoint()
    async with aiohttp.ClientSession(headers=headers) as session:
        for index, id_ in enumerate(tqdm(app_lists, desc='doing')):
            data_request = dict()
            data_request['request'] = 'appdetails'
            data_request['appid'] = str(id_)
            is_request_correct = check_request(data_request)
            data_request = fix_request(data_request)
            if is_request_correct:
                task = asyncio.get_event_loop().create_task(
                    download_for_app_id(session, url, id_, params=data_request))
                tasks.append(task)
            else:
                print("Incorrect request: download is cancelled.", id_)
            if len(tasks) == 200 or index >= app_size - 1:
                for task in tasks:
                    info = await task
                    if not info:
                        continue
                    else:
                        with open(output_file,'a+') as f:
                            f.write(json.dumps(info) + '\n')
                tasks = []

        for task in tasks:
            info = await task
            if not info:
                continue
            else:
                with open(output_file, 'a+') as f:
                    f.write(json.dumps(info) + '\n')
        tasks = []

if __name__ == "__main__":
    app_lists = get_data('./app_ids.txt')
    # for root,dirs,files in os.walk('./data'):
    #     got_apps = [int(file.split('_')[1].split('.')[0]) for file in files]
    # app_lists = [i for i in app_lists if i not in got_apps]
    print('need num',len(app_lists))
    asyncio.get_event_loop().run_until_complete(demo(app_lists,'app_detail_2.json'))
