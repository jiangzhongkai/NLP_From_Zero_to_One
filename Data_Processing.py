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

      def read_file(self,txt_file):

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
          f=self.read_file(txt_file)
          docs_xmls=f.split('<doc>\n')
          for doc in docs_xmls:
              if doc:
                  category,content=self.extract_class_content(doc)
                  self.write_file(category,content)

      #将数据集进行分割成训练集、测试集、验证集
      def save_file(self,dirname,target_dir):

          train_file=open(target_dir+"/new_train.txt",'w',encoding='utf-8')
          test_file=open(target_dir+"/new_test.txt",'w',encoding='utf-8')
          val_file=open(target_dir+'model_dataset/new_val.txt','w',encoding='utf-8')

          start=time.time()
          for category in os.listdir(dirname):
              cat_file=os.path.join(dirname,category)
              fp=open(cat_file,'rb')
              count=0
              for line in fp.readlines():
                  category=line.decode(encoding='utf-8',errors='ignore')[:len(str(category))]
                  content=line.decode(encoding='utf-8',errors='ignore')[len(str(category)):]
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
              print("finished",category)
          end=time.time()
          train_file.close()
          test_file.close()
          val_file.close()
          print("dataset has been splited,it costs:{:<.9f}".format(end-start))


if __name__=='__main__':

    dp=DP()
    # start=time.time()
    #
    # for file in os.listdir('dataset/'):
    #     file_path=os.path.join('dataset/',file)
    #     dp.category_data(file_path)
    # end=time.time()
    # print("data processing has been finished,it costs:{}secs.".format(str(end-start)))



















