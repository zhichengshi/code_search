import xml.etree.ElementTree as ET
import re
import os


def split_token(token):  # 分割叶子节点
    pattern = re.compile(r"[A-Z]{1}[a-z]+")
    result = pattern.findall(token)
    for item in result:
        token = token.replace(item, "")

    pattern2 = re.compile(r"[A-Z]+|[a-z]+")
    result += pattern2.findall(token)

    results = []
    for token in result:
        if len(token) > 1:
            results.append(token.lower())
    return results


def seq2id(desc, vocab):
    tokens = desc.split()
    desc_tokens = []
    for token in tokens:
        desc_tokens += split_token(token)
    

    return [vocab[token].index for token in desc_tokens if token in vocab]


def xml2astSeq(xml, vocab, language):  # 完成的代码预处理过程，源码 =>> 子树序列（子树接节点为index）

    def split_token(token):  # 分割叶子节点
        pattern = re.compile(r"[A-Z]{1}[a-z]+")
        result = pattern.findall(token)
        for item in result:
            token = token.replace(item, "")

        pattern2 = re.compile(r"[A-Z]+|[a-z]+")
        result += pattern2.findall(token)

        results = []
        for token in result:
            if len(token) > 1:
                results.append(token.lower())
        return results
    

    # 增加指向双亲节点的索引
    class treeNode:
        def __init__(self, parent, ele):
            if parent != None:
                self.parent = parent
                self.ele = ele
            else:
                self.parent = None
                self.ele = ele

    # 将叶子节点和中间节点使用同一种数据结构表示
    # 如果叶子节点可以分割，则分割叶子节点
    def transform(root):
        if root.text != None:
            # split leaf node and all subnodes are inserted into the ast
            tokens = split_token(root.text)
            for token in tokens:
                root.append(ET.Element(token))
        for child in root:
            transform(child)
        return root

    # 将ast中的节点以索引数字替代
    def tree_to_index(node, vocab):
        token = node.tag
        if token in vocab:
            result=[vocab[token].index]
        children = node.getchildren()
        for child in children:
            result.append(tree_to_index(child, vocab))
        return result


    # 深度优先遍历
    def dfs(root, list, parent=None):
        list.append(treeNode(parent, root))
        for node in root:
            dfs(node, list, root)
        return list

    '''
    step1: 预处理ast
    step2: 利用后续遍历获得子树节点
    step3: 处理子树，统一叶子节点以及中间节点的数据结构，若叶子节点是驼峰命名，则分割该叶子节点
    step4: 将树中的节点用索引替代
    '''
    split_nodes = set(["if", "while", "for", "function",'unit'])
    root =ET.fromstring(xml) 
    blocks = []
    dfs_node_sequence = dfs(root, [], None)
    for node in dfs_node_sequence:
        if node.ele.tag in split_nodes:
            blocks.append(node.ele)
            if node.parent != None:
                node.parent.remove(node.ele)

    blocks = [transform(block) for block in blocks]
    blocks = [tree_to_index(block, vocab) for block in blocks]
    return blocks
