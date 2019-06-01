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

def tf_eq():
    import tensorflow as tf
    import numpy as np

    tf_mu_prior = tf.constant([359.2917, 2.9117384, 0.69649607])
    cov_prior = np.array([[ 4.57899799e+03, 3.46347542e+01, -5.87668539e+00],\
            [ 3.46347542e+01, 1.39211206e+00, -5.56782887e-02], \
            [-5.87668539e+00, -5.56782887e-02, 2.48682567e-02]])
    tf_cov_prior = tf.constant([[ 4.57899799e+03, 3.46347542e+01, -5.87668539e+00],\
            [ 3.46347542e+01, 1.39211206e+00, -5.56782887e-02], \
            [-5.87668539e+00, -5.56782887e-02, 2.48682567e-02]])

    mu = tf.constant([[191.7485,   3.0277,  -0.5926],
            [181.4849,   2.8090,  -0.5407],
            [169.0526,   2.7422,  -0.3336],
            [180.0605,   2.8129,  -0.5387],
            [178.4771,   2.7660,  -0.5121],
            [195.7934,   3.2692,  -0.4653],
            [178.2992,   2.9080,  -0.3719],
            [175.5479,   2.8156,  -0.3457],
            [197.7952,   3.3001,  -0.4846],
            [194.2110,   3.1682,  -0.4701],
            [207.1334,   3.5591,  -0.5483],
            [177.6535,   2.7695,  -0.5204],
            [171.7478,   2.6651,  -0.4968],
            [169.5321,   2.7611,  -0.3269],
            [170.8998,   2.7862,  -0.3422],
            [211.9156,   3.5381,  -0.5800]])
    log_sigma_sq = tf.constant([[ 9.0258,  6.8842,  7.7061],
            [ 8.5538,  6.5004,  7.2741],
            [ 7.9989,  6.1633,  6.8041],
            [ 8.4929,  6.4884,  7.2242],
            [ 8.4249,  6.4168,  7.1545],
            [ 9.3091,  7.1848,  8.0120],
            [ 8.4183,  6.4290,  7.1961],
            [ 8.3437,  6.3523,  7.1306],
            [ 9.4027,  7.2599,  8.1027],
            [ 9.1355,  6.9722,  7.8086],
            [ 9.7605,  7.5904,  8.4266],
            [ 8.3907,  6.4119,  7.1336],
            [ 8.1242,  6.2188,  6.8793],
            [ 8.0184,  6.1509,  6.8341],
            [ 8.0810,  6.2386,  6.8772],
            [10.0104,  7.6790,  8.6467]])

    est_deviation = tf.subtract(tf_mu_prior, mu)
    cov_inverse = tf.matrix_inverse(tf_cov_prior)
    multi = tf.matmul( est_deviation, cov_inverse )

    log_det = np.log(np.linalg.det(cov_prior))

    sum_log_sigma = tf.reduce_sum(log_sigma_sq, 1)

    prod = tf.multiply( multi , tf.subtract(tf_mu_prior, mu) )
    su = tf.reduce_sum( prod, 1)

    exp_sgm = tf.exp(log_sigma_sq)
    diagonal = tf.matrix_diag(exp_sgm)

    all_traces = tf.trace( tf.scan(lambda a, x: tf.matmul(cov_inverse, x), diagonal ) )


    res = 0.5 * (all_traces
              + su
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

    results = [5814.085,  42248.52,   26438.46,   40205.72,   37500.113,  88262.99, \
      39071.496,  36592.613,  96629.555,  72023.28,  133582.67,   36730.746, \
      28507.848,  27232.748,  28441.168, 166324.81 ]

    print(sum(results)/16)
    #result = sess.run(e)
    #print("result ", result)


def check_view_mem():
    import torch
    a = torch.range(1, 16)
    b = a.view(4,4)
    a[0] = 23241
    print("a ", a)
    print("b ", b)

def check_diag():
    import torch
    print(torch.diag(torch.tensor([[8.1694, 7.8408, 8.0167], \
        [8.0776, 7.6264, 7.7638], \
        [7.2113, 6.6834, 7.1644],\
        [7.7860, 7.3877, 7.2154], \
        [7.8292, 7.3716, 7.7888], \
        [7.9325, 7.3160, 7.5938], \
        [7.4097, 7.0693, 7.2829], \
        [7.7946, 7.5154, 7.4965], \
        [7.2637, 6.9454, 7.0966], \
        [5.0692, 4.6383, 5.0361],\
        [8.2278, 8.1330, 7.9296], \
        [8.1595, 7.8885, 7.9317], \
        [7.5027, 6.9702, 7.4304], \
        [7.6518, 7.1061, 7.4969], \
        [7.5052, 7.1716, 7.3598], \
        [7.3951, 6.8756, 7.1188]])))


    print(torch.diag(torch.tensor([8.1694, 7.8408, 8.0167])))

    print(torch.diagonal(torch.tensor([[[ 2.8713e-02,  3.9204e-02, -5.5449e-03], \
         [-4.3051e-02,  4.4728e-02, -5.6602e-02], \
         [-1.9045e-02,  2.4839e-02,  2.2435e-02]], \

        [[ 2.9750e-02,  1.4640e-02,  1.6687e-02], \
         [-4.8086e-02,  3.3391e-02, -6.9261e-02], \
         [-2.4214e-02,  2.9605e-02,  4.6598e-02]], \

        [[ 2.7773e-02,  4.9777e-02, -6.2052e-04], \
         [-4.2726e-02,  2.6955e-02, -5.5824e-02], \
         [-2.4828e-02,  3.1523e-02,  4.1672e-02]], \

        [[ 1.9660e-06,  3.4006e-02,  1.4058e-02], \
         [-4.6490e-02,  4.9198e-02, -4.7358e-02], \
         [ 2.5900e-03,  2.6393e-02,  3.1483e-02]]]), dim1= -2 , dim2 = -1 ))

def try_concat():
    import numpy as np
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    print(np.vstack((a,b)))

try_concat()
