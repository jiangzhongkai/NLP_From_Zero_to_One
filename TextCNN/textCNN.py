"""-*- coding: utf-8 -*"""
"""
利用搜狗实验室的语料数据进行TextCNN的模型搭建
"""
import tensorflow as tf
import json

#先读取json文件中的相关配置参数
f=open("../Config.json",'r')
Config=json.load(f)

"""
整个网络结构如下：
    输入层->嵌入层->卷积层->池化层->全连接层->激活层->输出层
"""
class TextCNN(object):

    def __init__(self):

         self.X=tf.placeholder(dtype=tf.int32,shape=[None,Config['sequence_length']],name='X')
         self.Y=tf.placeholder(dtype=tf.float32,shape=[None,Config["num_classes"]],name='Y')
         self.drop_out=tf.placeholder(tf.float32)
         self.cnn_model()


    def cnn_model(self):

        with tf.name_scope("embedding_layer"):
            embdeeing=tf.get_variable(name="embedding_layer",shape=[Config['vocab_size'],Config['embedding_dim']])
            embdeeing_input=tf.nn.embedding_lookup(embdeeing,self.X)

        with tf.name_scope('conv_layer'):
            conv=tf.layers.conv1d(inputs=embdeeing_input,filters=Config['filter_nums'],kernel_size=Config["kernel_size"])

        with tf.name_scope("max_pooling"):
            max_pool=tf.nn.max_pool(conv)

        with tf.name_scope("fully_connected_layer"):
            fc=tf.layers.dense(max_pool,units=Config['hidden_nums'])

        with tf.name_scope("dropOut"):
            fc=tf.nn.dropout(fc,keep_prob=Config["dropout"])

        with tf.name_scope("relu"):
            fc=tf.nn.relu(fc)

        with tf.name_scope("output_layer"):
            self.logits=tf.layers.dense(fc,Config['num_classes'])
            self.y_pred_cls=tf.argmax(tf.nn.softmax(self.logits),1)

        with tf.name_scope("optimizer"):
            self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=self.Y))
            self.opt=tf.train.AdamOptimizer(Config["learning_rate"]).minimize(self.cost)

        with tf.name_scope("accuracy"):
            correct=tf.equal(tf.argmax(self.Y,1),self.y_pred_cls)
            accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

        return accuracy

    def train_model(self):
        pass
    def test_model(self):
        pass






