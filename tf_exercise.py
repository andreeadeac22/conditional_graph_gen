import tensorflow as tf

vec = tf.random_uniform(shape=(100,3,)) #log_sigma_sq
tf_cov_prior = tf.random_uniform(shape=(3,3,))

after_scan =  tf.scan(lambda a, x: tf.matmul(tf.matrix_inverse(tf_cov_prior), x), tf.matrix_diag(tf.exp(vec)) )

tr = tf.trace(after_scan)

sess = tf.Session()
print(sess.run(tr))
