# Steam-optimal_distinctiveness-anly
## 项目简介
论文题目：电子游戏产品差异化战略对绩效的影响：用户认知信息和情感信息的调节作用

此仓库为该毕设相关代码

包括 

Steam游戏信息、游戏评论、游戏用户打标等数据的爬虫代码

UDA、情感细粒度模型、情感极性模型训练代码

回归数据的处理与分析代码

## 数据爬虫

### Steam游戏信息

代码目录 craw_data_code / craw_steam_game_info

需要：
1. VPN

#### 1. 获取每个二级类别下的游戏列表

```
18 async with session.get(url,proxy='http://127.0.0.1:7890',timeout=10) as context:

修改自己的vpn端口

执行代码 python3 get_subcat.py

可以在目录下获得 cat_dict.json
```

#### 2. 获取每个游戏的Steam游戏信息


```
speed_up.py

91 proxy = 'http://127.0.0.1:7890'

修改自己的vpn端口

执行代码 python3 Topsell.py

可以在目录下获得 cat_app_info.json
```

### Steam游戏用户评论数据

代码目录 craw_data_code / get_cooment_data

需要：
1. 将代码上传至外网服务器执行
2. 外网ip池

#### 1. 准备需要爬虫的游戏id列表

在同目录下准备 app_lists.txt

其中每个游戏id一行

#### 2. 开始爬虫

```
craw_comment.py

修改 Proxy类 中的 get_one_proxy 方法以匹配自己的ip获得方式
默认使用ipidea，每次获得多个ip，单次返回的ip数量越多越好

执行代码 python3 Topsell.py

获得 data文件夹 包含所有游戏的评论数据
```

### metacritic用户评论情感数据

代码目录 craw_data_code / craw_meta_sentiment

该数据主要用于作为训练情感极性模型的标注数据

```
分别执行代码

python3 get_game_list.py

python3 get_comment.py

得到 comment_result.json 数据
```

## 模型训练

### Label2vec

代码目录 train_model / Label2vec

#### 1. 数据准备

1. 评论文件夹/data
2. cat_app_info.json

放在统一目录下

#### 2. 生成标签-文本训练语料

```
执行代码

python3 gen_w2v_corpus.py

得到语料文件 corpus.json
```

#### 3. 训练模型

```
执行代码

python3 train_doc2vec.py

得到label2vec模型文件 /doc2vec_model
```

### 情感模型训练

#### 无监督数据增强

代码目录 train_model / UDA / nlpaugmaster

转译增强damo aug.py

#### UDA训练

代码目录 train_model / uda_code

细粒度情感分析数据 SBAB / union_data.json

```
1. 修改main.py 中的文件参数配置对应监督数据、无监督数据以及测试数据

"sup_data_dir": "data/sup_train_data.json",
"unsup_data_dir": "data/unsup_data_list_aug.json",
"eval_data_dir": "data/sup_test_data.json",

2. 根据数据格式修改load_data.py中 SupDataset类 UnsupDataset类 EvalDataset类

3. 按需求修改main.py配置参数

4. 执行代码

accelerate config 设置训练环境

accelerate launch main.py 开始训练

得到模型文件
```

## 数据处理

代码目录 / reg_data_process


### 属性关键词字典
```
fin_core_word.ipynb 游戏属性关键词召回Ddamo

本研究属性-关键词字典见 core_word.json
```

### 数据处理

```
分别执行以下代码

python3 process.py 筛选有效的评论数据并预测情感极性

python3 process_2.py 评论属性关键词匹配以及属性情感分析

python3 build_control.py 建立控制变量

python3 deploy_info2cat.py 按二级分类对游戏进行分组

python3 compute_embedding.py 相关主效应、调节变量建立

reg.ipynb 回归分析damo
```





