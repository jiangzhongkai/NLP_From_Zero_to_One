"""-*- coding: utf-8 -*-"""
"""
fasttext:2016年facebook开源的一种词向量计算和文本分类的工具，训练精度与深度网络相媲美，训练时间却比深度网络块许多数量级
"""
import tensorflow as tf
import numpy as np

tf.flags.DEFINE_string("epoch",100,'the numbers of epoch')
tf.flags.DEFINE_bool()
