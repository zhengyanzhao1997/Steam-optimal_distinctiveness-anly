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

import copy
import datetime
import json
import pathlib
import time
import aiohttp
import asyncio
import requests
import os
from tqdm import tqdm

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
            id_lists.append(json.loads(i))
    return id_lists


async def request_func(session,url,params):
    for tries in range(10):
        try:
            proxy = next(proxy_pool)
            async with session.get(url,proxy=proxy,timeout=10,params=params) as context:
                json_ = await context.json()
                return context.status,json_
        except Exception as e:
            if tries < 9:
                continue
            print('detail Error: {}'.format(e))
            return None,None

def parse_app_id(app_id):
    # Objective: return an app_id as an integer, no matter whether the input app_id is a string or an integer.
    try:
        return int(str(app_id).strip())
    except ValueError:
        return None


def load_review_dict(app_id):
    review_data_filename = get_output_filename(app_id)

    try:
        with open(review_data_filename, "r", encoding="utf8") as in_json_file:
            review_dict = json.load(in_json_file)

        # Compatibility with data downloaded with previous versions of steamreviews:
        if "cursors" not in review_dict.keys():
            review_dict["cursors"] = dict()
    except FileNotFoundError:
        review_dict = dict()
        review_dict["reviews"] = dict()
        review_dict["query_summary"] = get_dummy_query_summary()
        review_dict["cursors"] = dict()

    return review_dict


def get_data_path():
    # Objective: return the path to the directory where reviews are stored.

    data_path = "data/"

    # Reference of the following line: https://stackoverflow.com/a/14364249
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

    return data_path


def get_steam_api_url():
    # Objective: return the url of Steam API for reviews.

    return "https://store.steampowered.com/appreviews/"


def get_output_filename(app_id):
    return get_data_path() + "review_" + str(app_id) + ".json"


def get_dummy_query_summary():
    query_summary = dict()
    query_summary["total_reviews"] = -1

    return query_summary

def get_default_request_parameters(chosen_request_params=None):
    # Objective: return a dict of default paramters for a request to Steam API.
    #
    # References:
    #   https://partner.steamgames.com/doc/store/getreviews
    #   https://partner.steamgames.com/doc/store/localization#supported_languages
    #   https://gist.github.com/adambuczek/95906b0c899c5311daeac515f740bf33

    default_request_parameters = {
        "json": "1",
        "language": "all",  # API language code e.g. english or schinese
        "filter": "recent",  # To work with 'start_offset', 'filter' has to be set to either recent or updated, not all.
        "review_type": "all",  # e.g. positive or negative
        "purchase_type": "all",  # e.g. steam or non_steam_purchase
        "num_per_page": "100",  # default is 20, maximum is 100
    }

    if chosen_request_params is not None:
        for element in chosen_request_params:
            default_request_parameters[element] = chosen_request_params[element]

    return default_request_parameters

def get_request(app_id, chosen_request_params=None):
    request = dict(get_default_request_parameters(chosen_request_params))
    request["appids"] = str(app_id)
    return request


async def download_reviews_for_app_id_with_offset(session, app_id, cursor="*", chosen_request_params=None):

    req_data = get_request(app_id, chosen_request_params)
    req_data["cursor"] = str(cursor)

    status_code,result = await request_func(session, get_steam_api_url() + req_data["appids"], req_data)

    if status_code == 200:
        result = result
    else:
        result = {"success": 0}
        print(
            "Faulty response status code = {} for appID = {} and cursor = {}".format(
                status_code, app_id, cursor
            )
        )

    success_flag = bool(result["success"] == 1)

    try:
        downloaded_reviews = result["reviews"]
        query_summary = result["query_summary"]
        next_cursor = result["cursor"]
    except KeyError:
        success_flag = False
        downloaded_reviews = []
        query_summary = get_dummy_query_summary()
        next_cursor = cursor

    return success_flag, downloaded_reviews, query_summary, next_cursor


async def download_reviews_for_app_id(
    session,
    app_id,
    chosen_request_params=None,
    start_cursor="*",  # this could be useful to resume a failed download of older reviews
    verbose=False,
):
    review_dict = load_review_dict(app_id)
    num_reviews = None

    cursor = start_cursor
    new_reviews = []
    new_review_ids = set()
    vaild_flag = True
    while True:
        if verbose:
            print(chosen_request_params)

        (
            success_flag,
            downloaded_reviews,
            query_summary,
            cursor,
        ) = await download_reviews_for_app_id_with_offset(
            session, app_id, '*', chosen_request_params
        )

        delta_reviews = len(downloaded_reviews)
        if delta_reviews == 0:
            if verbose:
                print(
                    "Exiting the loop to query Steam API, because the arrive empty."
                )
            break

        if success_flag and delta_reviews > 0:

            new_reviews.extend(downloaded_reviews)

            downloaded_review_ids = [
                review["recommendationid"] for review in downloaded_reviews
            ]
            min_review_time = min([
                review["timestamp_created"] for review in downloaded_reviews])

            chosen_request_params['end_date'] = min_review_time - 1
            # Detect full redundancy in the latest downloaded reviews
            if new_review_ids.issuperset(downloaded_review_ids):
                if verbose:
                    print(
                        "Exiting the loop to query Steam API, because this request only returned redundant reviews."
                    )
                vaild_flag = False
                break
            else:
                new_review_ids = new_review_ids.union(downloaded_review_ids)
                print(len(new_review_ids))

        else:
            # if verbose:
            vaild_flag = False
            print(
                "Exiting the loop to query Steam API, because this request failed."
            )
            break

        if num_reviews is None:
            if "total_reviews" in query_summary:
                review_dict["query_summary"] = query_summary
                num_reviews = query_summary["total_reviews"]


    if vaild_flag:
        review_dict["cursors"][str(cursor)] = time.asctime()
        for review in new_reviews:
            review_id = review["recommendationid"]
            if review_id not in review_dict["reviews"].keys():
                review_dict["reviews"][review_id] = review['timestamp_created']

        review_num = len(review_dict["reviews"].keys())
        if num_reviews:
            print("[appID = {}] expected #reviews = {} #got = {}".format(app_id, num_reviews,review_num))
        with open(get_output_filename(app_id), "w") as f:
            f.write(json.dumps(review_dict) + "\n")
        return review_dict


async def demo(app_lists):
    tasks = []
    app_size = len(app_lists)
    async with aiohttp.ClientSession(headers=headers) as session:
        for index, app in enumerate(tqdm(app_lists, desc='doing')):
            game_id = app['game_id']
            started = app['start']
            ended = app['end']
            day_range = app['day_range']
            request_params = {"start_date":started,"end_date":ended,"day_range":day_range,"date_range_type":"include"}
            task = asyncio.get_event_loop().create_task(download_reviews_for_app_id(session,game_id,chosen_request_params=request_params))
            tasks.append(task)
            if len(tasks) == 100 or index >= app_size - 1:
                for task in tasks:
                    info = await task
                tasks = []

if __name__ == "__main__":

    input_data = get_data('./xrc_need_lists.json')
    print('need num',len(input_data))
    asyncio.get_event_loop().run_until_complete(demo(input_data))


