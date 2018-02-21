# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import tensorflow as tf

W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)

#W = tf.Variable([-1.0],tf.float32)
#b = tf.Variable([1.0],tf.float32)

x = tf.placeholder(tf.float32)

linear_model = W * x + b


# specific to linear
#sess = tf.Session()
#
#init = tf.global_variables_initializer()
#sess.run(init)
#
#print sess.run(linear_model,{x:[1,2,3,4]})


#calculate loss

y = tf.placeholder(tf.float32)

squaredelta = tf.square(linear_model-y)
loss = tf.reduce_sum(squaredelta)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

print sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]})


#understanding loss recursion

'''
w = .3 b = -.3
w = -1 b = 1
linear_model = W*x +b
             = -1*1 + 1 = 0
             = -1*2 + 1 = -1
             = -1*3 + 1 = -2
             = -1*4 + 1 = -3
             
'''

#Optimizer


optimize = tf.train.GradientDescentOptimizer(0.01)
train = optimize.minimize(loss)
init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)



for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    
print(sess.run([W,b]))






