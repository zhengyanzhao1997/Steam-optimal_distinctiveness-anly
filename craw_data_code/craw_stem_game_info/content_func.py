import re
from lxml import etree
import datetime


def clean_trn(str_: str) -> str:
    str_ = str_.replace('\t', '').replace('\r', '').replace('\n', '').replace('\\t', '').replace('\\r', '').replace(
        '\\n', '').strip()
    return str_


def check_blank(str_: str) -> str:
    if len(str_.replace(' ', '')) == 0:
        return ""
    else:
        return str_


def convert_month(str_: str):
    if 'coming soon' in str_.lower():
        return 'coming soon'
    str_ = str_.replace('Jan', '1')
    str_ = str_.replace('Feb', '2')
    str_ = str_.replace('Mar', '3')
    str_ = str_.replace('Apr', '4')
    str_ = str_.replace('May', '5')
    str_ = str_.replace('Jun', '6')
    str_ = str_.replace('Jul', '7')
    str_ = str_.replace('Aug', '8')
    str_ = str_.replace('Sep', '9')
    str_ = str_.replace('Oct', '10')
    str_ = str_.replace('Nov', '11')
    str_ = str_.replace('Dec', '12')
    try:
        date = datetime.datetime.strptime(str_, '%m %d, %Y')
        return str(date.year) + '/' + str(date.month) + '/' + str(date.day)
    except:
        try:
            date = datetime.datetime.strptime(str_, '%d %m, %Y')
            return str(date.year) + '/' + str(date.month) + '/' + str(date.day)
        except:
            try:
                date = datetime.datetime.strptime(str_, '%m %Y')
                return str(date.year) + '/' + str(date.month)
            except:
                try:
                    date = datetime.datetime.strptime(str_, '%Y')
                    return str(date.year)
                except:
                    #print('cant find date',str_)
                    return None


def get_sys_list(sys_text):
    new_text = []
    for i in sys_text:
        if i != 'Minimum:' and 'Minimum:' in i:
            new_text.append('Minimum:')
            new_text.extend([x.strip() for x in i.split('Minimum:') if x.strip() != ""])
        elif i != 'Recommended:' and 'Recommended:' in i:
            new_text.append('Recommended:')
            new_text.extend([x.strip() for x in i.split('Recommended:') if x.strip() != ""])
        else:
            new_text.append(i)
    sys_text = new_text
    sys_card = {'Minimum': {}, 'Recommended': {}}
    idx = 1
    rec_bool = 0
    no_min = 0
    find_rec = 0
    if sys_text[0] != 'Minimum:' and sys_text[0][-1] == ':':
        no_min = 1
        idx -= 1
    while idx < len(sys_text):
        if sys_text[idx] == 'Recommended:':
            rec_bool = 1
            break
        if sys_text[idx][-1] == ":":
            if len(sys_text) > 1 and idx == len(sys_text) - 1:
                break
            key = sys_text[idx][:-1]
            if (key in sys_card['Minimum'].keys() and no_min) or find_rec:
                find_rec = 1
                sys_card['Recommended'][key] = sys_text[idx + 1]
            else:
                sys_card['Minimum'][key] = sys_text[idx + 1]
            idx += 2
        else:
            sys_card['Minimum']['text'] = sys_text[idx]
            idx += 1

    if rec_bool and not no_min:
        idx += 1
        while idx < len(sys_text):
            if sys_text[idx][-1] == ":":
                if len(sys_text) > 1 and idx == len(sys_text) - 1:
                    break
                key = sys_text[idx][:-1]
                sys_card['Recommended'][key] = sys_text[idx + 1]
                idx += 2
            else:
                sys_card['Recommended']['text'] = sys_text[idx]
                idx += 1
    return sys_card


def get_onegame(content_text, pre_difined_item):
    app_id = pre_difined_item['game_id']
    app_name = pre_difined_item['name']
    tree_info = etree.HTML(content_text)
    game_title = tree_info.xpath('//*[@id="appHubAppName"]/text()')[0]
    info_card = tree_info.xpath('//*[@id="genresAndManufacturer"]//text()')
    info_card = [clean_trn(x) for x in info_card if clean_trn(x).replace(' ', '').replace(',', '') != ""]
    init_card = {}
    idx = 0
    key = None
    while idx < len(info_card):
        if info_card[idx][-1] == ":":
            key = info_card[idx][:-1]
            init_card[key] = []
        elif key:
            init_card[key].append(info_card[idx])
        idx += 1
    genre = []
    genre_count = 0
    if 'Genre' in init_card.keys():
        genre = init_card['Genre']
        genre_count = len(genre)

    developer = []
    developer_count = 0
    if 'Developer' in init_card.keys():
        developer = init_card['Developer']
        developer_count = len(developer)

    publisher = []
    publisher_count = 0
    if 'Publisher' in init_card.keys():
        publisher = init_card['Publisher']
        publisher_count = len(publisher)

    franchiser = []
    franchiser_count = 0
    if 'Franchise' in init_card.keys():
        franchiser = init_card['Franchise']
        franchiser_count = len(franchiser)

    release_date = None
    if 'Release Date' in init_card.keys():
        release_date = init_card['Release Date'][0]
        release_date = convert_month(release_date)
        if not release_date:
            release_date = init_card['Release Date'][0]


    usertag = tree_info.xpath('//*[@class="app_tag"]//text()')
    usertag = [clean_trn(x) for x in usertag]
    usertag_count = len(usertag)

    IF_DLC = 'This content requires the base game' in content_text

    description_short = tree_info.xpath('//*[@class="game_description_snippet"]//text()')
    if len(description_short) > 0:
        description_short = clean_trn(description_short[0])
    else:
        description_short = ""
    description_title = check_blank(
        ' '.join([clean_trn(x) for x in tree_info.xpath('//*[@id="game_area_description"]/h2/text()')])).strip()
    description_long = check_blank(
        ' '.join([clean_trn(x) for x in tree_info.xpath('//*[@id="game_area_description"]//text()')]))
    description_long = description_long.replace(description_title, "").strip()
    price_ = -1
    purchase = tree_info.xpath('//*[@class="game_purchase_action"]')
    if len(purchase) > 0:
        purchase = purchase[0].xpath('.//text()')
        price = [clean_trn(x) for x in purchase if clean_trn(x).replace(' ', '').replace(',', '') != ""]
        for i in range(len(price)):
            if price[i] == 'Add to Cart':
                break
            if '$' in price[i]:
                price_ = price[i].replace('$', '')
            if price[i] in ['Free to Play', 'Free', 'Play Game','Download'] or 'Free' in price[i]:
                price_ = 0
                break

    feature = tree_info.xpath('//*[@class="game_area_details_specs_ctn"]//text()')
    feature = [clean_trn(x) for x in feature if clean_trn(x).replace(' ', '').replace(',', '') != ""]
    feature_count = len(feature)

    learning_about = tree_info.xpath('//*[@class="game_area_details_specs_ctn learning_about"]')
    steam_learning = 0
    if len(learning_about) > 0:
        steam_learning = 1

    language = tree_info.xpath('//*[@id="languageTable"]/table//text()')
    language = [clean_trn(x) for x in language if clean_trn(x).replace(' ', '').replace(',', '').replace('✔', '') != ""]
    language = language[3:]
    language_count = len(language)

    rating = None
    rating_standard = None
    rating_icon = None
    if 'shared_game_rating' in content_text:
        rating_icon_ = tree_info.xpath('//*[@class="game_rating_icon"]//@src')
        if len(rating_icon_) > 0:
            rating_icon = rating_icon_[0]
        rating_ = tree_info.xpath('//*[@class="game_rating_descriptors"]/p/text()')
        if len(rating_) > 0:
            rating = [clean_trn(x) for x in rating_ if clean_trn(x) != ""]
        rating_standard = tree_info.xpath('//*[@class="game_rating_agency"]//text()')
        if len(rating_standard) > 0:
            rating_standard = rating_standard[0].split("Rating for:")[1].strip()

    award_bool = 0
    award = None
    award_ = tree_info.xpath('//*[@id="awardsTable"]')
    if len(award_) > 0:
        award_bool = 1
        banner_event = award_[0].xpath('.//*[@class="steamawards_app_banner_event"]')
        banner_context = award_[0].xpath('.//*[@class="steamawards_app_banner_contents"]')
        award = ""
        if len(banner_event) > 0:
            assert len(banner_event) == len(banner_context)
            for i,j in zip(banner_event,banner_context):
                itext = i.xpath('.//text()')
                jtext = j.xpath('.//text()')
                i = ' '.join([clean_trn(x) for x in itext if clean_trn(x).replace(' ','').replace(',','') != ""])
                j = ' '.join([clean_trn(x) for x in jtext if clean_trn(x).replace(' ','').replace(',','') != ""])
                award += i + ': ' + j + '; '
            award_text = award_[0].xpath('.//text()')
            award_text = [clean_trn(x) for x in award_text]
            find_end = 0
            for idx,x in enumerate(award_text):
                if x.replace(' ','').replace(',','') != "" and x not in award:
                    award_text = award_text[idx:]
                    find_end = 1
                    break
            if find_end:
                for idx,x in enumerate(award_text):
                    if x.replace(' ','').replace(',','') != "" and idx < len(award_) -1 and award_[idx+1].replace(' ','').replace(',','') == "":
                        award += x + '; '
                    elif x.replace(' ','').replace(',','') != "":
                        award += x
        else:
            award_text = award_[0].xpath('.//text()')
            award_ = [clean_trn(x) for x in award_text]
            for idx,x in enumerate(award_):
                if x.replace(' ','').replace(',','') != "" and idx < len(award_) -1 and award_[idx+1].replace(' ','').replace(',','') == "":
                    award += x + '; '
                elif x.replace(' ','').replace(',','') != "":
                    award += x

    metacritic = tree_info.xpath('//*[@id="game_area_metascore"]/div[1]/text()')
    if len(metacritic) > 0:
        metacritic = clean_trn(metacritic[0])
    else:
        metacritic = None

    curator = re.findall(r'<div class="no_curators_followed">(.*?)Curators have reviewed this product',
                         repr(content_text))
    if len(curator) > 0:
        curator = clean_trn(curator[0])
    else:
        curator = -1

    reviews = tree_info.xpath('//*[@id="userReviews"]/div//text()')
    reviews = [clean_trn(x) for x in reviews if clean_trn(x).replace(' ', '').replace(',', '') not in ["","*"]]
    reviews_dict = {}
    idx = 0
    while idx < len(reviews):
        if reviews[idx][-1] == ":":
            key = reviews[idx][:-1]
            reviews_dict[key] = []
        else:
            reviews_dict[key].append(reviews[idx])
        idx += 1

    recent_review_count = None
    recent_review_sum = None
    recent_review_detail = None
    recent_review_percent = None
    review_bool = 1
    review_not_enough_bool = 0
    review_count = None
    review_sum = None
    review_detail = None
    review_percent = None

    if 'Recent Reviews' in reviews_dict.keys():

        if len(reviews_dict['Recent Reviews']) > 2 and reviews_dict['Recent Reviews'][1].strip()[0] == '(' and reviews_dict['Recent Reviews'][1].strip()[-1] == ')':
            recent_review_count = int(reviews_dict['Recent Reviews'][1].replace('(', '').replace(')', '').replace(',', ''))
            recent_review_sum = reviews_dict['Recent Reviews'][0]
            recent_review_detail = reviews_dict['Recent Reviews'][2]
            recent_review_percent = re.findall(r'[0-9]+%', recent_review_detail)
            recent_review_percent = recent_review_percent[0]
        else:
            print(app_id)
            raise
        #
        # elif len(reviews_dict['Recent Reviews']) > 1 and len(re.findall(r'[0-9]+%', reviews_dict['Recent Reviews'][1])) > 0:
        #     recent_review_detail = reviews_dict['Recent Reviews'][1]
        #     recent_review_percent = re.findall(r'[0-9]+%', recent_review_detail)
        #     recent_review_sum = reviews_dict['Recent Reviews'][0]
        #     recent_review_count = re.findall(r' (\d+(?:[.,]\d+)*) ',recent_review_detail)
        #     recent_review_count = recent_review_count[0]

    if 'All Reviews' in reviews_dict.keys():

        if len(reviews_dict['All Reviews']) > 2 and reviews_dict['All Reviews'][1].strip()[0] == '(' and reviews_dict['All Reviews'][1].strip()[-1] == ')':
            review_count = int(reviews_dict['All Reviews'][1].replace('(', '').replace(')', '').replace(',', ''))
            review_sum = reviews_dict['All Reviews'][0]
            review_detail = reviews_dict['All Reviews'][2]
            review_percent = re.findall(r'[0-9]+%', review_detail)
            review_percent = review_percent[0]

        # elif len(reviews_dict['All Reviews']) > 1 and len(re.findall(r'[0-9]+%', reviews_dict['All Reviews'][1])) > 0:
        #     review_detail = reviews_dict['All Reviews'][1]
        #     review_percent = re.findall(r'[0-9]+%', review_detail)
        #     review_sum = reviews_dict['All Reviews'][0]
        #     review_count = re.findall(r' (\d+(?:[.,]\d+)*) ',review_detail)
        #     review_count = review_count[0]

        elif len(reviews_dict['All Reviews']) > 1 and reviews_dict['All Reviews'][1] == '- Need more user reviews to generate a score':
            review_count = re.findall(r'[0-9]+', reviews_dict['All Reviews'][0])[0]
            review_not_enough_bool = 1
        elif reviews_dict['All Reviews'][0] == 'No user reviews':
            review_bool = 0
        else:
            print(app_id)
            raise

    system_name_list = tree_info.xpath('//div[contains(@class,"sysreq_tab")]')
    system_name_list = [clean_trn(x.xpath('./text()')[0]) for x in system_name_list]
    system_name_list = [x.strip() for x in system_name_list if x.strip() != ""]
    system_num = len(system_name_list)
    system_req_list = tree_info.xpath('//div[contains(@class,"game_area_sys_req sysreq_content")]')
    system_req_list = [x.xpath('.//text()') for x in system_req_list]
    sys_list = []
    if system_num == 0 and len(system_req_list) == 0:
        pass
    elif system_num == 0:
        system_num += 1
        assert len(system_req_list) == 1
        sys_text = [clean_trn(x) for x in system_req_list[0] if
                    clean_trn(x).replace(' ', '').replace(',', '') != ""]
        if len(sys_text) > 0:
            sys_card = get_sys_list(sys_text)
            if 'OS' in sys_card['Minimum'].keys():
                sys_list.append({"system":sys_card['Minimum']['OS'],'Minimum':sys_card['Minimum'],'Recommended':sys_card['Recommended']})
            else:
                sys_list.append({"system": None, 'Minimum': sys_card['Minimum'],'Recommended': sys_card['Recommended']})
    else:
        assert system_num == len(system_req_list)
        for sys_name, sys_detail in zip(system_name_list, system_req_list):
            sys_detail = [clean_trn(x) for x in sys_detail if
                          clean_trn(x).replace(' ', '').replace(',', '') != ""]
            if len(sys_detail) > 0:
                sys_card = get_sys_list(sys_detail)
                sys_list.append({"system":sys_name, 'Minimum': sys_card['Minimum'],
                                 'Recommended': sys_card['Recommended']})

    DLC_bool = 0
    DLC_count = 0
    DLC_detail_list = []
    DLC = tree_info.xpath('//*[@class="gameDlcBlocks"]/a')
    DLC_expend = tree_info.xpath('//*[@id="game_area_dlc_expanded"]/a')
    if len(DLC) > 0:
        DLC_bool = 1
        if len(DLC_expend) > 0:
            DLC += DLC_expend
        DLC_count = len(DLC)
        for dlc in DLC:
            name_list = clean_trn(dlc.xpath('./*[@class="game_area_dlc_name"]//text()')[0])
            price_list = [clean_trn(x) for x in dlc.xpath('./*[@class="game_area_dlc_price"]//text()') if
                          clean_trn(x) != ""][-1].replace("$", "")
            DLC_detail_list.append((name_list, price_list))

    new_item_dict = {
        'game_id': app_id,
        'name': app_name,
        'request_status':'Available',
        'game_title': game_title,
        'IF_DLC': IF_DLC,
        'genre': genre,
        'genre_count': genre_count,
        'developer': developer,
        'developer_count': developer_count,
        'publisher': publisher,
        'publisher_count': publisher_count,
        'franchiser': franchiser,
        'franchiser_count': franchiser_count,
        'release_date': release_date,
        'usertag': usertag,
        'usertag_count': usertag_count,
        'description_short': description_short,
        'description_title': description_title,
        'description_long': description_long,
        'price': price_,
        'feature': feature,
        'feature_count': feature_count,
        'steam_learning': steam_learning,
        'language': language,
        'language_count': language_count,
        'rating': rating,
        'rating_standard': rating_standard,
        'rating_icon': rating_icon,
        'award_bool': award_bool,
        'award': award,
        'metacritic': metacritic,
        'curator': curator,
        'review_bool': review_bool,
        'review_not_enough_bool': review_not_enough_bool,
        'review_count': review_count,
        'review_sum': review_sum,
        'review_detail': review_detail,
        'review_percent': review_percent,
        'recent_review_count': recent_review_count,
        'recent_review_sum': recent_review_sum,
        'recent_review_detail': recent_review_detail,
        'recent_review_percent': recent_review_percent,
        'system_count': system_num,
        'sys_card': sys_list,
        'DLC_bool': DLC_bool,
        'DLC_count': DLC_count,
        'DLC_detail_list': DLC_detail_list, }
    return new_item_dict


def get_onepackage(content_text, pre_difined_item):
    app_id = pre_difined_item['game_id']
    app_name = pre_difined_item['name']
    tree_info = etree.HTML(content_text)
    game_title = tree_info.xpath('//*[@id="appHubAppName"]//text()')
    info_card = tree_info.xpath('//*[@class="details_block"]//p//text()')
    info_card = [clean_trn(x) for x in info_card if clean_trn(x).replace(' ', '').replace(',', '') != ""]
    init_card = {}
    idx = 0
    key = None
    while idx < len(info_card):
        if info_card[idx][-1] == ":":
            key = info_card[idx][:-1]
            init_card[key] = []
        elif key:
            init_card[key].append(info_card[idx])
        idx += 1

    developer = []
    developer_count = 0
    if 'Developer' in init_card.keys():
        developer = init_card['Developer']
        developer_count = len(developer)

    publisher = []
    publisher_count = 0
    if 'Publisher' in init_card.keys():
        publisher = init_card['Publisher']
        publisher_count = len(publisher)

    franchiser = []
    franchiser_count = 0
    if 'Franchise' in init_card.keys():
        franchiser = init_card['Franchise']
        franchiser_count = len(franchiser)

    language = []
    language_count = 0
    if 'Languages' in init_card.keys():
        language = [x.strip() for x in init_card['Languages'][0].split(',')]
        language_count = len(language)

    release_date = init_card['Release Date'][0]
    release_date = convert_month(release_date)

    description_title = check_blank(
        ' '.join([clean_trn(x) for x in tree_info.xpath('//*[@id="game_area_description"]/h2/text()')])).strip()
    description_long = check_blank(
        ' '.join([clean_trn(x) for x in tree_info.xpath('//*[@id="game_area_description"]//text()')]))
    description_long = description_long.replace(description_title, "").strip()

    price_ = -1
    purchase = tree_info.xpath('//*[@class="game_purchase_action"]')
    if len(purchase) > 0:
        purchase = purchase[0].xpath('.//text()')
        price = [clean_trn(x) for x in purchase if clean_trn(x).replace(' ', '').replace(',', '') != ""]
        for i in range(len(price)):
            if price[i] == 'Add to Cart':
                break
            if '$' in price[i]:
                price_ = price[i].replace('$', '')
            if price[i] in ['Free to Play', 'Free', 'Play Game'] or 'Free' in price[i]:
                price_ = 0
                break

    feature = tree_info.xpath('//*[@class="game_area_details_specs"]//text()')
    feature = [clean_trn(x) for x in feature]
    feature_count = len(feature)

    learning_about = tree_info.xpath('//*[@class="game_area_details_specs learning_about"]')
    steam_learning = 0
    if len(learning_about) > 0:
        steam_learning = 1

    new_item_dict = {
        'game_id': app_id,
        'name': app_name,
        'request_status': 'Package',
        'game_title': game_title,
        'developer': developer,
        'developer_count': developer_count,
        'publisher': publisher,
        'publisher_count': publisher_count,
        'franchiser': franchiser,
        'franchiser_count': franchiser_count,
        'release_date': release_date,
        'description_title': description_title,
        'description_long': description_long,
        'price': price_,
        'feature': feature,
        'feature_count': feature_count,
        'steam_learning': steam_learning,
        'language': language,
        'language_count': language_count}
    return new_item_dict