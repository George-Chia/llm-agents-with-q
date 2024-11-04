import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import re

# 加载JSON文件
with open('webshop/trajectories_test_llama31-0_T1.0_mcts_30iterations/1642.json', 'r') as f:
    data = json.load(f)


def parse_action(llm_output: str) -> str:
    llm_output = llm_output.strip()
    try:
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
    except:
        action = 'nothing'
    assert action is not None
    return action

# 递归解析JSON数据并构建图形
def add_nodes_edges(graph, node, parent=None):
    # 提取节点的关键属性
    node_name = f"{parse_action(node['state']['action'])}"
    graph.add_node(node_name, visits=node.get('visits', 0), value=node.get('value', 0))
    
    if parent:
        graph.add_edge(parent, node_name)
        
    # 递归处理子节点
    children = node.get('children', [])
    if children:  # 仅在 children 存在并非 None 时才遍历
        for child in children:
            add_nodes_edges(graph, child, node_name)

# 构建NetworkX图
G = nx.DiGraph()
add_nodes_edges(G, data)

# 绘制树状图
plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G)  # 使用 spring_layout 代替 graphviz_layout
pos = graphviz_layout(G, prog="dot")  # 使用 Graphviz 布局

# 绘制节点
nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8, font_weight="bold")

# 绘制节点标签
node_labels = nx.get_node_attributes(G, 'visits')
nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="black")

plt.title("Tree Structure Visualization")
plt.savefig('webshop/tree.png')
