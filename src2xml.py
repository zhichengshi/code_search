'''
将代码转化成ast,用xml的格式保存起来
'''
import os
import xml.etree.ElementTree as ET
import pandas as pd
import signal

def set_timeout(timeout):
    def wrap(func):
        def handle(signum, frame):  
            raise RuntimeError

        def to_do(*args):
            try:
                signal.signal(signal.SIGALRM, handle)  # 设置信号和回调函数
                signal.alarm(timeout)  # 设置 num 秒的闹钟
                r = func(*args)
                signal.alarm(0)  # 关闭闹钟
                return r
            except RuntimeError as e:
                return "time out"
        return to_do

    return wrap


class src2xml():
    def __init__(self, root, language, part):
        self.root = root
        self.part = part
        self.language = language

    @set_timeout(2)  # 限时 2 秒
    def parse(self,code):  # 生成代码抽象语法树 src => ast
        code = code.replace("\"", " \\\"")
        cmd = "srcml --text=\" "+code + "\" -l "+language
        xml = os.popen(cmd).read()
        xml = xml.replace("xmlns=\"http://www.srcML.org/srcML/src\"", "")
        return xml

    def transform(self):
        raw_data_path = os.path.join(self.root, self.language, self.part+".csv")
        raw_data_pd = pd.read_csv(raw_data_path)
        xmls = []
        codes = []
        descs = []
        for _, row in raw_data_pd.iterrows():
            xmls.append(self.parse(row['code']))
            codes.append(row['code'])
            descs.append(row['query'])

        df = pd.DataFrame()
        df['code'] = codes
        df['xml'] = xmls
        df['description'] = descs

        #dump data
        dump_path = os.path.join(self.root, self.language, 'xml', self.part+".json")
        df.to_json(dump_path)

if __name__ == "__main__":
    root='dataset'
    language='C#'
    part='test'
    procedure=src2xml(root,language,part)
    procedure.transform()
    
