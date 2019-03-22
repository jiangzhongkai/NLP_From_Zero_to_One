"""-*- coding: utf-8 -*-"""
import pandas as pd
import jieba
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
import time

""""
example:
    <doc>
    <url>http://sports.sohu.com/s2008/2917/s256966143/</url>
    <docno>006b8e1746c29427-71013306c0bb3300</docno>
    <contenttitle></contenttitle>
    <content></content>
    </doc>
"""

DIR_NAME="processed_dataset/"

class DP(object):

      def __init__(self):

          pass

      def _read_file(self,txt_file):

          return open(txt_file,'rb').read().decode('gbk',errors='ignore')

      def extract_class_content(self,doc):

          url=doc.split('<url>')[1].split('</url>')[0]
          content=doc.split('<content>')[1].split('</content>')[0]
          # category=re.findall(r"http://(.*?).sohu.com/",url)
          category=url.split("//")[1].split('.')[0]
          return category,content

      def write_file(self,category,content):
          path=os.path.join(DIR_NAME,category)
          f=open(path,'a',encoding='utf-8')
          f.write(category+'\t'+content+'\n')
          f.close()


      def category_data(self,txt_file):
          f=self._read_file(txt_file)
          docs_xmls=f.split('<doc>\n')
          for doc in docs_xmls:
              if doc:
                  category,content=self.extract_class_content(doc)
                  self.write_file(category,content)


      def split_data_into_train_test_val(self,dirname,target_dirname):
          """
          :param dirname: 源目录文件
          :param target_dirname: 目标目录
          :return: 按一定比例分割完成的数据集
          """
          train_file=open(target_dirname+'/new_train.txt','w',encoding='gbk',errors='ignore')
          test_file=open(target_dirname+'/new_test.txt','w',encoding='gbk',errors='ignore')
          val_file=open(target_dirname+'/new_val.txt','w',encoding='gbk',errors='ignore')

          #从原目录文件中读取数据
          start=time.time()
          for category in os.listdir(dirname):
              #对文件内容进行读取
              file=os.path.join(dirname,category)
              fp=open(file,'r',encoding='utf-8',errors='ignore')
              count=0
              for line in fp.readlines():
                  category=line[:len(str(category))]
                  content=line[len(str(category)):]
                  if category and content:
                      if count<500:
                          train_file.write(category+'\t'+content+'\n')
                      elif count<700:
                          test_file.write(category+'\t'+content+'\n')
                      elif count<850:
                          val_file.write(category+'\t'+content+'\n')
                  else:
                      break
                  count+=1
              print("finished {0}".format(category))
          end=time.time()
          print("all category have been finished,total cost {0}sec".format(str(end-start)))


if __name__=='__main__':

    dp=DP()
    start=time.time()

    for file in os.listdir('dataset/'):
        file_path=os.path.join('dataset/',file)
        dp.category_data(file_path)
    end=time.time()
    print("data processing has been finished,it costs:{}secs.".format(str(end-start)))
















