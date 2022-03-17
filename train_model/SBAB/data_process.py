import json
import csv
import json
import xmltodict

####acl-14-short-data

start_token = '<ts>'
end_token = '<te>'

def build_insert(n):
    return start_token + ' ' + n + end_token

data_acl = []
for acl_path in ['./data/acl-14-short-data/train.raw','./data/acl-14-short-data/test.raw']:
    group = []
    item = []
    with open(acl_path,'r') as f:
        for index,i in enumerate(f):
            i = i.replace('\n','')
            item += [i]
            if (index + 1) % 3 == 0:
                group.append(item)
                item = []

    for item in group:
        target = build_insert(item[1])
        assert '$T$' in item[0]
        text = item[0].replace('$T$',target)
        label = int(item[2])
        data_acl.append((text,label))

def read_from_tsv(file_path: str,) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf,fieldnames=['text','label_0','label_1','label_2','label_3','label_4'],dialect='tsv_dialect')
        datas = []
        for row in reader:
            datas.append(row)
    csv.unregister_dialect('tsv_dialect')
    return datas


ACOS_data = []

for path in ['./data/ACOS/ACOS-main/data/Laptop-ACOS/laptop_quad_test.tsv',
             './data/ACOS/ACOS-main/data/Laptop-ACOS/laptop_quad_dev.tsv',
             './data/ACOS/ACOS-main/data/Laptop-ACOS/laptop_quad_train.tsv',
             './data/ACOS/ACOS-main/data/Restaurant-ACOS/rest16_quad_test.tsv',
             './data/ACOS/ACOS-main/data/Restaurant-ACOS/rest16_quad_dev.tsv',
             './data/ACOS/ACOS-main/data/Restaurant-ACOS/rest16_quad_train.tsv']:
    data = read_from_tsv(path)
    for d in data:
        text = d['text']
        text_split = text.split()
        add_taget = []
        for i in range(5):
            target = d['label_%s'%i]
            if target:
                target = target.split()
                s_e = target[0].split(',')
                if int(s_e[0]) == -1:
                    continue
                label = target[2]
                label_s = int(s_e[0])
                if label_s in add_taget:
                    continue
                add_taget.append(label_s)
                label_e = int(s_e[1])
                n = ' '.join(text_split[label_s:label_e])
                assert n in text
                re_n = build_insert(n)
                text_temp = text.replace(n,re_n)
                label = int(label)
                ACOS_data.append((text_temp,label-1))

mams_data = []

def xml_to_json(xml_path):
    f = open(xml_path)
    data = f.read()
    xml_parse = xmltodict.parse(data)
    json_str = json.dumps(xml_parse, indent=1)
    data = json.loads(json_str)
    return data


emotion_dict = {'negative':-1,'neutral':0,'positive':1}
for path in ['./data/MAMS/test.xml','./data/MAMS/train.xml','./data/MAMS/val.xml',]:
    data = xml_to_json(path)['sentences']['sentence']
    for d in data:
        text = d['text']
        for target in d['aspectTerms']['aspectTerm']:
            term = target['@term']
            assert term in text
            re_n = build_insert(term)
            text_temp = text.replace(term,re_n)
            label = emotion_dict[target['@polarity']]
            mams_data.append((text_temp,label))


for path in [
             './data/SemEval14/Laptop_Train_v2.xml',
             './data/SemEval14/Restaurants_Train_v2.xml']:
    data = xml_to_json(path)['sentences']['sentence']
    for d in data:
        text = d['text']
        if 'aspectTerms' in d.keys():
            if isinstance(d['aspectTerms']['aspectTerm'],list):
                for target in d['aspectTerms']['aspectTerm']:
                    term = target['@term']
                    assert term in text
                    re_n = build_insert(term)
                    text_temp = text.replace(term,re_n)
                    if target['@polarity'] in emotion_dict.keys():
                        label = emotion_dict[target['@polarity']]
                        mams_data.append((text_temp, label))
            else:
                target = d['aspectTerms']['aspectTerm']
                term = target['@term']
                assert term in text
                re_n = build_insert(term)
                text_temp = text.replace(term, re_n)
                if target['@polarity'] in emotion_dict.keys():
                    label = emotion_dict[target['@polarity']]
                    mams_data.append((text_temp, label))


data_union = data_acl + ACOS_data  + mams_data
print(len(data_union))

with open('./union_data.json','w') as f:
    for d in data_union:
        item = json.dumps({'text':d[0],'label':d[1]})
        f.write(item + '\n')