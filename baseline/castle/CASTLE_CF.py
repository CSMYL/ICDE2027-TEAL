import numpy as np

np.set_printoptions(suppress=True)
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from sklearn.metrics import mean_squared_error
import time

np.set_printoptions(linewidth=np.inf)


class CASTLE(object):
    def __init__(self, num_train, lr=None, batch_size=32, num_inputs=1, num_outputs=1,
                 w_threshold=0.3, n_hidden=32, hidden_layers=2, ckpt_file='tmp.ckpt',
                 standardize=True, reg_lambda=None, reg_beta=None, DAG_min=0.5, y_dim=3):

        self.y_dim = y_dim
        self.w_threshold = w_threshold
        self.DAG_min = DAG_min
        if lr is None:
            self.learning_rate = 0.001
        else:
            self.learning_rate = lr

        if reg_lambda is None:
            self.reg_lambda = 1.
        else:
            self.reg_lambda = reg_lambda

        if reg_beta is None:
            self.reg_beta = 1
        else:
            self.reg_beta = reg_beta

        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.X = tf.placeholder("float", [None, self.num_inputs])
        self.y = tf.placeholder("float", [None, self.y_dim])

        self.rho = tf.placeholder("float", [1, 1])
        self.alpha = tf.placeholder("float", [1, 1])
        self.keep_prob = tf.placeholder("float")
        self.Lambda = tf.placeholder("float")
        self.noise = tf.placeholder("float")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        self.count = 0
        self.max_steps = 200
        self.saves = 50
        self.patience = 15
        self.metric = mean_squared_error

        self.sample = tf.placeholder(tf.int32, [self.num_inputs])

        seed = 1
        self.weights = {}
        self.biases = {}

        for i in range(self.num_inputs):
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
            self.weights['w_h0_' + str(i)] = tf.Variable(initializer([self.num_inputs, self.n_hidden]))
            self.weights['out_' + str(i)] = tf.Variable(tf.random_normal([self.n_hidden, self.num_outputs], seed=seed))

        for i in range(self.num_inputs):
            self.biases['b_h0_' + str(i)] = tf.Variable(tf.random_normal([self.n_hidden], seed=seed) * 0.01)
            self.biases['out_' + str(i)] = tf.Variable(tf.random_normal([self.num_outputs], seed=seed))

        self.weights.update({
            'w_h1': tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
        })

        self.biases.update({
            'b_h1': tf.Variable(tf.random_normal([self.n_hidden]))
        })

        self.hidden_h0 = {}
        self.hidden_h1 = {}
        self.layer_1 = {}
        self.layer_1_dropout = {}
        self.out_layer = {}

        self.Out_0 = []

        self.mask = {}
        self.activation = tf.nn.relu

        for i in range(self.num_inputs):
            if i >= self.num_inputs - self.y_dim:
                # Label prediction nodes, mask all label inputs (last y_dim columns)
                mask_vec = np.ones((self.num_inputs,), dtype=np.float32)
                mask_vec[-self.y_dim:] = 0.0  # Mask all label inputs
            else:
                # Normal feature nodes, mask only themselves
                mask_vec = np.ones((self.num_inputs,), dtype=np.float32)
                mask_vec[i] = 0.0  # Mask themselves

            # Extend to (num_inputs, n_hidden)
            mask_matrix = np.tile(mask_vec[:, np.newaxis], (1, self.n_hidden))  # shape: (num_inputs, n_hidden)
            self.mask[str(i)] = tf.constant(mask_matrix, dtype=tf.float32)

            # self.weights['w_h0_' + str(i)] = self.weights['w_h0_' + str(i)] * self.mask[str(i)]
            #
            # self.hidden_h0['nn_' + str(i)] = self.activation(
            #     tf.add(tf.matmul(self.X, self.weights['w_h0_' + str(i)]), self.biases['b_h0_' + str(i)]))

            # W_masked = self.weights['w_h0_' + str(i)] * self.mask[str(i)]
            masked_weight = tf.stop_gradient(self.mask[str(i)]) * self.weights['w_h0_' + str(i)]
            # self.hidden_h0['nn_' + str(i)] = self.activation(
            #     tf.add(tf.matmul(self.X, W_masked), self.biases['b_h0_' + str(i)])
            # )
            self.hidden_h0['nn_' + str(i)] = self.activation(
                tf.add(tf.matmul(self.X, masked_weight), self.biases['b_h0_' + str(i)])
            )

            self.hidden_h1['nn_' + str(i)] = self.activation(
                tf.add(tf.matmul(self.hidden_h0['nn_' + str(i)], self.weights['w_h1']), self.biases['b_h1']))
            self.out_layer['nn_' + str(i)] = tf.matmul(self.hidden_h1['nn_' + str(i)], self.weights['out_' + str(i)]) + \
                                             self.biases['out_' + str(i)]
            self.Out_0.append(self.out_layer['nn_' + str(i)])

        # Concatenate all the constructed features
        self.Out = tf.concat(self.Out_0, axis=1)
        self.optimizer_subset = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # Here, nn_0 output is considered as y output, and self.out_layer contains the entire reconstruction
        # self.supervised_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.out_layer['nn_0'] - self.y), axis=1), axis=0)
        # Here, replace with softmax+CEloss
        # Take the last self.y_dim outputs from self.out_layer
        self.logits_list = [
            self.out_layer['nn_{}'.format(self.num_inputs - i - 1)]
            for i in range(self.y_dim)
        ]

        # Concatenate to (batch_size, y_dim)
        # self.logits = tf.concat(self.logits_list[::-1], axis=1) + 0.5  # Note [::-1] to ensure ascending order
        self.logits = tf.concat(self.logits_list[::-1], axis=1)
        # Calculate softmax cross-entropy
        self.supervised_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y,  # y is already one-hot
                logits=self.logits
            )
        )


        self.regularization_loss = 0

        self.W_0 = []
        for i in range(self.num_inputs):
            self.W_0.append(
                tf.math.sqrt(tf.reduce_sum(tf.square(self.weights['w_h0_' + str(i)]), axis=1, keepdims=True)))

        self.W = tf.concat(self.W_0, axis=1)

        # truncated power series
        d = tf.cast(self.X.shape[1], tf.float32)
        coff = 1.0
        Z = tf.multiply(self.W, self.W)

        dag_l = tf.cast(d, tf.float32)

        Z_in = tf.eye(d)
        for i in range(1, 10):
            Z_in = tf.matmul(Z_in, Z)

            dag_l += 1. / coff * tf.linalg.trace(Z_in)
            coff = coff * (i + 1)

        self.h = dag_l - tf.cast(d, tf.float32)

        # Residuals
        self.R = self.X - self.Out
        # Average reconstruction loss
        self.average_loss = 0.5 / num_train * tf.reduce_sum(tf.square(self.R))

        # group lasso
        L1_loss = 0.0
        for i in range(self.num_inputs):
            w_1 = tf.slice(self.weights['w_h0_' + str(i)], [0, 0], [i, -1])
            w_2 = tf.slice(self.weights['w_h0_' + str(i)], [i + 1, 0], [-1, -1])
            L1_loss += tf.reduce_sum(tf.norm(w_1, axis=1)) + tf.reduce_sum(tf.norm(w_2, axis=1))

        # Divide the residual into untrain and train subset
        _, subset_R = tf.dynamic_partition(tf.transpose(self.R), partitions=self.sample, num_partitions=2)
        subset_R = tf.transpose(subset_R)

        # Combine all the loss
        self.mse_loss_subset = tf.cast(self.num_inputs, tf.float32) / tf.cast(tf.reduce_sum(self.sample),
                                                                              tf.float32) * tf.reduce_sum(
            tf.square(subset_R))
        self.regularization_loss_subset = self.mse_loss_subset + self.reg_beta * L1_loss + 0.5 * self.rho * self.h * self.h + self.alpha * self.h

        # Add in supervised loss
        self.regularization_loss_subset += self.Lambda * self.rho * self.supervised_loss
        self.loss_op_dag = self.optimizer_subset.minimize(self.regularization_loss_subset)

        self.loss_op_supervised = self.optimizer_subset.minimize(self.supervised_loss + self.regularization_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.tmp = ckpt_file


    def __del__(self):
        tf.reset_default_graph()
        print("Destructor Called... Cleaning up")
        self.sess.close()
        del self.sess

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    def fit(self, X, y, num_nodes, X_val, y_val, X_test, y_test):

        from random import sample
        rho_i = np.array([[1.0]])
        alpha_i = np.array([[1.0]])

        best = 1e9
        best_value = 1e9
        for step in range(1, self.max_steps):
            epoch_start_time = time.time()
            h_value, loss, logits_val, out_vals = self.sess.run(
                [self.h, self.supervised_loss, self.logits, self.Out],
                feed_dict={
                    self.X: X,
                    self.y: y,
                    self.keep_prob: 1,
                    self.rho: rho_i,
                    self.alpha: alpha_i,
                    self.is_train: True,
                    self.noise: 0
                }
            )
            
            # print(f"Logits (first 3):\n{logits_val[:3]}")
            # print(f"Out Layer (first 3):\n{out_vals[:3, -self.y_dim:]}")

            for step1 in range(1, (X.shape[0] // self.batch_size) + 1):

                idxs = random.sample(range(X.shape[0]), self.batch_size)
                batch_x = X[idxs]
                batch_y = y[idxs]

                one_hot_sample = [0] * self.num_inputs
                subset_ = sample(range(self.num_inputs), num_nodes)
                for j in subset_:
                    one_hot_sample[j] = 1
                self.sess.run(self.loss_op_dag,
                              feed_dict={self.X: batch_x, self.y: batch_y, self.sample: one_hot_sample,
                                         self.keep_prob: 1, self.rho: rho_i, self.alpha: alpha_i,
                                         self.Lambda: self.reg_lambda, self.is_train: True, self.noise: 0})
                # logits_val, y_v, ce_loss_val, loss = self.sess.run(
                #     [self.logits, self.y,
                #      tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits), self.supervised_loss],
                #     feed_dict={
                #         self.X: batch_x,
                #         self.y: batch_y,
                #         self.sample: one_hot_sample,
                #         self.keep_prob: 1,
                #         self.rho: rho_i,
                #         self.alpha: alpha_i,
                #         self.Lambda: self.reg_lambda,
                #         self.is_train: True,
                #         self.noise: 0
                #     }
                # )
                # print("Logits:\n", logits_val[:3])
                # print("Labels:\n", y_v[:3])
                # print("CrossEntropy loss vector:\n", ce_loss_val[:3])
                # print("Any NaN in logits?", np.isnan(logits_val).any())
                # print("Any NaN in labels?", np.isnan(y_v).any())
                # print("Any NaN in CE loss?", np.isnan(ce_loss_val).any())
                # print("Any logits > 1e2?", np.max(np.abs(logits_val)) > 1e2)
                # print("self.loss", loss)
            
            val_loss = self.val_loss(X_val, y_val)
            if val_loss < best_value:
                best_value = val_loss
            h_value, loss = self.sess.run([self.h, self.supervised_loss],
                                          feed_dict={self.X: X, self.y: y, self.keep_prob: 1, self.rho: rho_i,
                                                     self.alpha: alpha_i, self.is_train: True, self.noise: 0})
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f"Step {step}, Loss= {loss:.4f}, h_value: {h_value:.4f}, Epoch time: {epoch_time:.4f} seconds")
            
            if step >= self.saves:
                try:
                    if val_loss < best:
                        best = val_loss
                        self.saver.save(self.sess, self.tmp)
                        print("Saving model")
                        self.count = 0
                    else:
                        self.count += 1
                except:
                    print("Error caught in calculation")

            if self.count > self.patience:
                print("Early stopping")
                break

        self.saver.restore(self.sess, self.tmp)
        W_est = self.sess.run(self.W,
                              feed_dict={self.X: X, self.y: y, self.keep_prob: 1, self.rho: rho_i, self.alpha: alpha_i,
                                         self.is_train: True, self.noise: 0})
        W_est[np.abs(W_est) < self.w_threshold] = 0

    def val_loss(self, X, y):
        if len(y.shape) < 2:
            y = np.expand_dims(y, -1)
        from random import sample
        one_hot_sample = [0] * self.num_inputs

        # use all values for validation
        subset_ = sample(range(self.num_inputs), self.num_inputs)
        for j in subset_:
            one_hot_sample[j] = 1

        return self.sess.run(self.supervised_loss,
                             feed_dict={self.X: X, self.y: y, self.sample: one_hot_sample, self.keep_prob: 1,
                                        self.rho: np.array([[1.0]]),
                                        self.alpha: np.array([[0.0]]), self.Lambda: self.reg_lambda,
                                        self.is_train: False, self.noise: 0})

    def pred(self, X):
        y_pred = self.sess.run(
            tf.argmax(self.logits, axis=1),  # Use self.logits, already includes +0.5
            feed_dict={self.X: X, self.keep_prob: 1, self.is_train: False, self.noise: 0}
        )
        return y_pred

    def get_weights(self, X, y):
        return self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob: 1, self.rho: np.array([[1.0]]),
                                                self.alpha: np.array([[0.0]]), self.is_train: False, self.noise: 0})

    def pred_W(self, X, y=None):
        if y is None:
            y = np.zeros((X.shape[0], self.y_dim))  # dummy one-hot y

        W_est = self.sess.run(self.W, feed_dict={
            self.X: X,
            self.y: y,
            self.keep_prob: 1,
            self.rho: np.array([[1.0]]),
            self.alpha: np.array([[0.0]]),
            self.is_train: False,
            self.noise: 0
        })
        return np.round_(W_est, decimals=3)

