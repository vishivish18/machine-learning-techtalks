# -*- coding: utf-8 -*-

import tensorflow as tf
a = tf.constant(5.0)
b = tf.constant(6.0)

c = a*b
print(c)


file  = tf.summary.FileWriter('/test/',sess.graph)
sess = tf.Session()


print(sess.run(c))