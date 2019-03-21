"""-*- coding: utf-8 -*-"""
"""学习NLP第一个demo"""

import tensorflow as tf
import numpy as np

""""
主要步骤总结如下：
     1.导入相应的库
     2.对语料进行处理:
        1)主要是进行去掉停用词，标点符号
        2）分词
        3）去重
        4）建立单词到索引的映射，相当于词向量
        5）建立索引到单词的映射
    3.设置模型参数
    4.获取批量数据
    5.进行变量和占位符的定义
    5.模型搭建-->损失函数、优化算法
    6.训练模型
    7.保存模型
    8.测试模型
"""

tf.reset_default_graph()

#语料的处理
sentences=['i like dog','i love coffee','i hate milk','i love you','wang hate jiang']

#分词
word_list=' '.join(sentences).split()
#去重
word_list=list(set(word_list))
#建立单词到索引的字典
word_dict={w:i for i,w in enumerate(word_list)}
#建立索引到单词的字典
num_dict={i:w for i,w in enumerate(word_list)}
#字典的数目
n_class=len(word_dict)

#设置模型的参数
n_step=2
n_hidden=2

#获取批量数据
def make_batch(sentences):
    input_batch=[]
    target_batch=[]

    for sen in sentences:

        word=sen.split()
        input=[word_dict[n] for n in  word[:-1]]
        target=word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])
    return input_batch,target_batch

#模型的搭建、
X=tf.placeholder(tf.float32,[None,n_step,n_class])
Y=tf.placeholder(dtype=tf.float32,shape=[None,n_class])

#修改数据输入格式
input=tf.reshape(X,shape=[-1,n_class*n_step])
#输入层到隐藏层的权重，以及偏置
H=tf.Variable(tf.random_normal([n_step*n_class,n_hidden]))
d=tf.Variable(tf.random_normal([n_hidden]))
#隐藏层到输出层的权重和偏置
U=tf.Variable(tf.random_normal([n_hidden,n_class]))
b=tf.Variable(tf.random_normal([n_class]))

tanh=tf.nn.tanh(d+tf.matmul(input,H))

#模型的输出
model=tf.matmul(tanh,U)+b

#损失函数
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=Y))
optimizer=tf.train.AdamOptimizer(0.001).minimize(cost)
#目标函数
prediction=tf.arg_max(model,1)

#初始化变量
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

input_batch,output_batch=make_batch(sentences)

for epoch in range(10000):
    _,loss=sess.run([optimizer,cost],feed_dict={X:input_batch,Y:output_batch})
    if (epoch+1)%10==0:
        print("epoch:{},loss:{}".format(str(epoch),loss))
    #

predict=sess.run([prediction],feed_dict={X:input_batch})

print(predict)

#Test
input=[sen.split()[:2] for sen in sentences]
print([sen.split()[:2] for sen in sentences],'->',[num_dict[n] for n in predict[0]])