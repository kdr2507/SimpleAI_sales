import tensorflow as tf

xData = [1, 2, 3, 4, 5, 6, 7]  # 하루 노동시간
yData = [10000, 25000, 45000, 70000, 10000, 150000, 180000]  # 하루 매출액

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1], -100, 100))
b = tf.Variable(tf.random_uniform([1], -100, 100))  # y절편

a = tf.Variable(0.01)  # 경사하강에서 어느 거리만큼 이동할것인지를 결정
optimizer = tf.train.GradientDescentOptimizer(a)  # 경사하강 라이브러리


sales = W * x + b   # => 노동시간에 따른 하루 매출액
cost_function = tf.reduce_mean(tf.square(sales - y))
train = optimizer.minimize(cost_function)


work_time = (sales-b) / W     # => 하루 매출액에 따른 노동시간
work_time_function = tf.reduce_mean(tf.square(work_time - x))
work_time_train = optimizer.minimize(work_time_function)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(5001):
    sess.run(train, feed_dict={x: xData, y: yData})
    sess.run(work_time_train, feed_dict={x: xData, sales: yData})
    if(i % 500 == 0):
        print("트레이닝 횟수 : ", i, sess.run(cost_function, feed_dict={
            x: xData, y: yData}), "예상 매출액 계산중 : ", sess.run(sales, feed_dict={x: [10]}))

print("노동시간에 따른 예상 매출액 : ", sess.run(sales[0], feed_dict={x: [10]}), "원")
print("하루 매출액에 따른 예상 노동시간 :", sess.run(
    work_time[0], feed_dict={sales: [250000]}), "시간")
