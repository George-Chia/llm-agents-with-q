
from node import *

root = Node(state=None, question='aaaaa')

def modify(node):
    node.children = 'dsddd'
    print(node.children)

    
print(root.children)
modify(root)
print(root.children)