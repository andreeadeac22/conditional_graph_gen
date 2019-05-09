def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: int(x == s), allowable_set))


#print(onek_encoding_unk(4, [0,1,2,3,4,5]) + onek_encoding_unk('C', ['O', 'S', 'unk']))


def append_lists():
    list1 = ['av', 'vsdgs']
    list2 = ['fwaw', 'wkjrfn']
    list = list1 + list2
    print(list)

import tensorflow as tf
import numpy as np

tf_mu_prior = tf.constant([359.2917, 2.9117384, 0.69649607])
cov_prior = np.array([[ 4.57899799e+03, 3.46347542e+01, -5.87668539e+00],\
        [ 3.46347542e+01, 1.39211206e+00, -5.56782887e-02], \
        [-5.87668539e+00, -5.56782887e-02, 2.48682567e-02]])
tf_cov_prior = tf.constant([[ 4.57899799e+03, 3.46347542e+01, -5.87668539e+00],\
        [ 3.46347542e+01, 1.39211206e+00, -5.56782887e-02], \
        [-5.87668539e+00, -5.56782887e-02, 2.48682567e-02]])

mu = tf.constant ( \
    [[ 0.0364, -0.0283, -0.0021], \
        [ 0.0455, -0.0328, -0.0165], \
        [ 0.0466,  0.0216,  0.0262], \
        [ 0.0470, -0.0190,  0.0042]] )

log_sigma_sq = tf.constant( \
    [[-3.6918e-03,  1.8677e-02,  1.9783e-02], \
        [-9.1692e-04,  8.6478e-02,  1.6718e-02], \
        [ 6.0552e-02, -3.2075e-02, -8.3384e-03], \
        [-1.6187e-02,  6.3404e-02, -2.2113e-05]] )


est_deviation = tf.subtract(tf_mu_prior, mu)
cov_inverse = tf.matrix_inverse(tf_cov_prior)
multi = tf.matmul( est_deviation, cov_inverse )

log_det = np.log(np.linalg.det(cov_prior))

sum_log_sigma = tf.reduce_sum(log_sigma_sq, 1)

prod = tf.multiply( multi , tf.subtract(tf_mu_prior, mu) )
sum = tf.reduce_sum( prod, 1)

exp_sgm = tf.exp(log_sigma_sq)
diagonal = tf.matrix_diag(exp_sgm)

all_traces = tf.trace( tf.scan(lambda a, x: tf.matmul(cov_inverse, x), diagonal ) )


res = 0.5 * (all_traces
          + sum
          - float(3) + log_det - sum_log_sigma )

sess = tf.Session()

# Execute the graph and store the value that `e` represents in `result`.
with sess.as_default():
  #print("est_deviation ", est_deviation.eval())
  print("cov_inverse ", cov_inverse.eval())
  #print("multi ", multi.eval())

  #print("prod ", prod.eval())
  #print("sum ", sum.eval())
  #print("exp_sgm ", exp_sgm.eval())
  #print("diagonal ", diagonal.eval())
  print("log_det ", log_det)
  print("all_traces ", all_traces.eval())
  print("sum_log_sigma ", sum_log_sigma.eval())
  print("res ", res.eval())


#result = sess.run(e)
#print("result ", result)
