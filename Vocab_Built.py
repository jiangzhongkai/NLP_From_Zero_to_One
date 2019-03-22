"""-*- coding: utf-8 -*-
 @DateTime   : 2019/3/21 16:09
 @author     : Peter_Bonnie
"""
from Data_Processing import *
from collections import Counter
from keras.utils import to_categorical
from keras.preprocessing import sequence

"""构建词汇表类"""
class VocabBuilt(object):

    def __init__(self,train_path,vocab_path,vocab_size,dp,remove_list):
        """
        :arguments:
            train_path:训练集的路径
            vocab_path:单词表的路径
            vocab_size:单词表的大小
            dp:   数据处理类的对象
        :return:
            vocab.txt
        """
        self.train_path=train_path
        self.vocab_path=vocab_path
        self.vocab_size=vocab_size
        self.dp=dp
        self.remove_list=remove_list

    def built_vocab(self):
        """构造单词表"""

        """定义一个内部函数，多次调用"""
        def _read_file(data_path):

            if not os.path.exists(data_path):
                print("{0} not found.".format(data_path))
            else:
                f=open(data_path,'rb')
                return f.read().decode(encoding='utf-8',errors='ignore')

        data_train=_read_file(self.train_path)
        all_data=[]
        for content in data_train:
            if content not in self.remove_list:
                all_data.extend(content)

        "计算有多少个单词"
        counter=Counter(all_data)
        counter_pairs=counter.most_common(self.vocab_size-1)

        words,_=list(zip(*counter_pairs))
        words=['<PAD>']+list(words)
        if os.path.isfile(self.vocab_path):
            os.remove(self.vocab_path)
        open(self.vocab_path,mode='w',encoding='gbk',errors='ignore').write('\n'.join(words)+'\n')

    """建立单词到索引的映射"""
    def word_2_idx(self):

        if not os.path.exists(self.vocab_path):
            print("{0} not found.".format(self.vocab_path))
            exit()
        else:
            with open(self.vocab_path) as f:
                words=[_.strip() for _ in f.readlines()]
                word_2_id=dict(zip(words,range(len(words))))

            return words,word_2_id

    """建立类别与id的映射"""
    def category_2_idx(self,category_list):

        category_2_id=dict(zip(category_list,range(len(category_list))))

        return category_list,category_2_id

    """主要是处理正文和类别"""
    def process_file(self,filename,category_2_id=None,word_2_id=None,max_length=None):

        #先是读取数据
        if not os.path.exists(filename):
            print("{0} not found.".format(filename))
        else:
            data_id,label_id=[],[]
            fp = open(filename, 'r', encoding='utf-8', errors='ignore')
            for line in fp.readlines():
                labels=line.split('\t')[0]
                content=line.split('\t')[1:]
                for i in range(len(content)):
                    data_id.append([word_2_id[x] for x in content[i] if x in word_2_id])
                    label_id.append(category_2_id[label_id[i]])
            x_pad=sequence.pad_sequences(data_id,max_length)
            y_pad=to_categorical(label_id,num_classes=len(category_2_id))

            return x_pad,y_pad

if __name__=='__main__':

    dp=DP()
    #剔除文本中的类名
    remove_list=[]
    for dir in os.listdir("processed_dataset/"):
        remove_list.append(dir)

    vb=VocabBuilt(train_path='model_dataset/new_train.txt',vocab_path='model_dataset/vocab.txt',vocab_size=5000,dp=dp,remove_list=remove_list)

    # vb.built_vocab()
    vb.process_file("model_dataset/new_train.txt")

    # words,word_2_id=vb.word_2_idx()
    # print(word_2_id)

















