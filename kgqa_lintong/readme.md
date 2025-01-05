## 1 环境准备
1. 在原llmp的环境下，还需要安装SPARQLWrapper、rank_bm25、sentence_transformers
2. 调用知识图谱 freebase 的地址在 hotpot_tog/tog/freebase_func.py中修改 SPARQLPATH = "http://10.107.7.80:8890/sparql" 

## 2 修改说明
1. 增加 node 的属性 topic_entity ,对于根节点，topic_entity 是问题的关键词，对于非根节点，topic_entity 是父节点关键词的经图谱扩展剪枝后的实体。
2. 为了便于把节点的topic_entity 传递到prune 方法，增加了env的属性 env.node,具体部分在mcts.py/generate_new_states_fastchat_conv和 wrappers.py 的 LoggingWrapper类
3. 主要在 wikienv.py 中增加了 prune 方法，根据父节点传入的 topic_entity,往下扩展并更新topic_entity

4. 每次剪枝后扩展的完整的三元组序列（也就是从根节点到该节点的完整路径）放在observation中

5. 杂项：
修改了hotpotqa.py 中 tokenizer 的地址；
为了适配图谱问答的数据集，对 WikiEnv 和 HotPotQAWrapper有一点修改

6. tog文件夹说明：
* tog/prompt_list.py 中是图谱搜索的提示词；
* tog/freebase_func.py 中是调用freebase的函数、以及一些知识图谱的搜索、剪枝函数；
* tog/utils.py 中是支撑freebase_func的函数；
* tog/clinet.py 连接图谱服务器；
* tog/search.py 中是原来写的搜索wiki的代码，这里没有用；


## 3 运行说明
1. 测试的数据集是 data/cwq.json, trajectories_test_gpt_simple_1iterations/51.json 是一个运行结果示例
2. run.py 中的 --max_depth为图谱的搜索深度
