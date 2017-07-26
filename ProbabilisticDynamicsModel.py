from __future__ import division
import time
import tensorflow as tf
import numpy as np
import math

from ops import *
from utils import *


class ProbabilisticDynamicsModel:
    def build_model(self):
        raise NotImplementedError
    def train(self, traj, cfg):
        raise NotImplementedError
    def predict(self, x_t, u_t):
        raise NotImplementedError

def nonlinear_mlp(x, u, num_mc_samples, hidden_layer_sizes, dropout_prob, is_test, N_train):
    x_dim = x.get_shape().as_list()[1]
    u_dim = u.get_shape().as_list()[1]

    model_input = tf.concat([x, u], axis=1)
    model_input = tf.expand_dims(model_input, axis=1)
    model_input = tf.tile(model_input, [1, num_mc_samples, 1])

    log_var_x_next = tf.get_variable("log_var_x_next", [1, 1, x_dim], tf.float32,
                          tf.constant_initializer(-5.0))
    dropout_reg = 0 #1e-6 #2*tf.sqrt(tf.reduce_sum(tf.exp(log_var_x_next)))/N_train

    # model_input has shape (K,N,M), corresponding to the K samples input
    # to the model, N = self.num_mc_samples copies of each input for MC
    # epistemic uncertainty calculation, and M is the x_dim + u_dim
    z, reg_loss = mlp_with_dropout(model_input, hidden_layer_sizes, dropout_prob, is_test)
    # z, reg_loss = mlp_with_concrete_dropout(model_input, hidden_layer_sizes, is_test, dropout_reg=dropout_reg)

    x_reshaped = tf.expand_dims(x, axis=1)
    x_next_hat, reg_loss2 = dense_with_dropout(z, output_size=x_dim,
                            prob=dropout_prob, is_test=is_test, name="fc_mean")
    # x_next_hat, reg_loss2 = dense_with_concrete_dropout(z, output_size=x_dim,
    #                                                 is_test=is_test, dropout_reg=dropout_reg,
    #                                                 name="fc_mean")
    # x_next_hat += x_reshaped

    reg_loss += reg_loss2

    return (x_next_hat, log_var_x_next, reg_loss)

def locally_linear_mlp(x, u, num_mc_samples, hidden_layer_sizes, dropout_prob, is_test, N_train):
    x_dim = x.get_shape().as_list()[1]
    u_dim = u.get_shape().as_list()[1]
    x_tiled = tf.tile(tf.expand_dims(x, axis=1), [1, num_mc_samples, 1])
    u_tiled = tf.tile(tf.expand_dims(u, axis=1), [1, num_mc_samples, 1])
    model_input = tf.concat([x_tiled, u_tiled], axis=2)

    # Variance is independent of input (homoscedastic)
    log_var_x_next = tf.get_variable("log_var_x_next", [1, 1, x_dim], tf.float32,
                          tf.constant_initializer(-5.0))

    dropout_reg = 0.02*tf.sqrt(tf.reduce_sum(tf.exp(log_var_x_next)))/N_train

    z, reg_loss = mlp_with_dropout(model_input, hidden_layer_sizes, dropout_prob, is_test)
    # z, reg_loss = mlp_with_concrete_dropout(model_input, hidden_layer_sizes, is_test, dropout_reg=dropout_reg)

    A_dim = x_dim**2
    B_dim = x_dim*u_dim
    A_flat, reg_loss_A = dense_with_dropout(z, output_size=A_dim,
                           prob=dropout_prob, is_test=is_test, name="fc_A")
    B_flat, reg_loss_B = dense_with_dropout(z, output_size=B_dim,
                           prob=dropout_prob, is_test=is_test, name="fc_B")
    # A_flat, reg_loss_A = dense_with_concrete_dropout(z,
    #                        dropout_reg=dropout_reg, output_size=A_dim,
    #                        is_test=is_test, name="fc_A")
    # B_flat, reg_loss_B = dense_with_concrete_dropout(z,
    #                        dropout_reg=dropout_reg, output_size=B_dim,
    #                        is_test=is_test, name="fc_B")

    A = tf.reshape(A_flat, [-1, x_dim, x_dim], name="A_into_matrix")
    B = tf.reshape(B_flat, [-1, x_dim, u_dim], name="B_into_matrix")

    x_reshaped = tf.expand_dims( tf.reshape(x_tiled, [-1, x_dim], name="x_into_matrix"), axis=-1)
    u_reshaped = tf.expand_dims( tf.reshape(u_tiled, [-1, u_dim], name="u_into_matrix"), axis=-1)
    x_next_hat = x_reshaped + A @ x_reshaped + B @ u_reshaped
    x_next_hat = tf.squeeze(x_next_hat, axis=-1)
    x_next_hat = tf.reshape(x_next_hat, [-1, num_mc_samples, x_dim], name="result_to_batch")


    reg_loss += reg_loss_A + reg_loss_B
    return (x_next_hat, log_var_x_next, reg_loss)


MLP_DM_cfg = {
    "lr": 2e-3,
    "beta1": 0.9,
    "batch_size": 100,
    "n_epochs": 100,
    "store_val": True
}

class MLPDynamicsModel(ProbabilisticDynamicsModel):
    def __init__(self, sess, x_dim, u_dim, hidden_layer_sizes=[64, 64, 64],
                 dropout_prob=0.2, num_mc_samples=50, filename="64_64_64",
                 writer_path="MLP_DM"):
        self.sess = sess
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_prob = dropout_prob
        self.num_mc_samples = num_mc_samples
        self.filename = filename
        self.writer_path = writer_path
        self.counter = 1
        self.pdm_op = locally_linear_mlp

    def build_model(self):
        self.x_ = tf.placeholder(tf.float32, (None, self.x_dim))
        self.u_ = tf.placeholder(tf.float32, (None, self.u_dim))
        self.x_next_ = tf.placeholder(tf.float32, (None, self.x_dim))
        self.is_test = tf.placeholder_with_default(False, [])
        self.N_train = tf.placeholder(tf.float32, [])

        with tf.variable_scope("MLP_DM"):
            self.x_next_hat, self.log_var_x_next, self.reg_loss = self.pdm_op(self.x_, self.u_,
                                                                              self.num_mc_samples,
                                                                              self.hidden_layer_sizes,
                                                                              self.dropout_prob,
                                                                              self.is_test,
                                                                              self.N_train)

        with tf.variable_scope("Losses_and_Metrics"):
            model_target = tf.expand_dims(self.x_next_, axis=1)
            eps = 1e-7
            mahalanobis_dist = mahalanobis_distance(self.x_next_hat, model_target, tf.exp(self.log_var_x_next))
            mahalanobis_loss = tf.reduce_mean(mahalanobis_dist)
            with tf.variable_scope("variance_regularization"):
                var_reg_loss = tf.reduce_mean( tf.reduce_sum( self.log_var_x_next, axis=2 ) )
            with tf.variable_scope("l2_loss"):
                l2_loss = tf.reduce_mean( tf.reduce_sum( (self.x_next_hat - model_target)**2, axis=2 ) )

            self.loss = mahalanobis_loss + var_reg_loss + self.reg_loss

            # Various quantities of interest
            with tf.variable_scope("prediction_error"):
                error = tf.reduce_mean(tf.reduce_mean(self.x_next_hat - model_target, axis=2), axis=1)

            aleatoric_unc = tf.reduce_mean(tf.exp(self.log_var_x_next))
            ep_unc_of_mean = tf.reduce_mean(epistemic_unc(self.x_next_hat))
            #ep_unc_of_var = tf.reduce_mean(epistemic_unc(self.log_var_x_next))


            # Summaries
            error_sum = tf.summary.histogram("prediction_error", error)
            mahalanobis_dist_sum = tf.summary.histogram("mahalanobis_dist", mahalanobis_dist)
            maha_loss_sum = tf.summary.scalar("mahalanobis_loss", mahalanobis_loss)
            var_reg_loss_sum = tf.summary.scalar("var_reg_loss", var_reg_loss)
            l2_loss_sum = tf.summary.scalar("l2_loss", l2_loss)
            total_loss_sum = tf.summary.scalar("total_loss", self.loss)
            reg_loss_sum = tf.summary.scalar("regularization_loss", self.reg_loss)
            aleatoric_unc_sum = tf.summary.scalar("aleatoric_unc", aleatoric_unc)
            ep_unc_of_mean_sum = tf.summary.scalar("epistemic_unc_of_mean", ep_unc_of_mean)
            #ep_unc_of_var_sum = tf.summary.scalar("epistemic_unc_of_var", ep_unc_of_var)
            self.summary = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.writer_path + "/" + self.filename + '-train', self.sess.graph)
            self.val_writer = tf.summary.FileWriter(self.writer_path + "/" + self.filename + '-val', self.sess.graph)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(max_to_keep=2)

    def train(self, transitions, cfg=MLP_DM_cfg):
        optim = tf.train.AdamOptimizer(cfg["lr"], beta1=cfg["beta1"]
                                       ).minimize(self.loss, var_list=self.t_vars)

        tf.global_variables_initializer().run()

        train_trans, val_trans = split_train_val(transitions)

        # Assemble training inputs and targets
        N = len(train_trans["x"])
        x = np.reshape(np.array(train_trans["x"]), [N, -1])
        u = np.reshape(np.array(train_trans["u"]), [N, -1])
        x_next = np.reshape(np.array(train_trans["x_next"]), [N, -1])

        N_val = len(val_trans["x"])
        x_val = np.reshape(np.array(val_trans["x"]), [N_val, -1])
        u_val = np.reshape(np.array(val_trans["u"]), [N_val, -1])
        x_next_val = np.reshape(np.array(val_trans["x_next"]), [N_val, -1])

        print("Training with", N, "input/target pairs.")

        start_time = time.time()

        for epoch in range(cfg["n_epochs"]):
            num_batches = N // cfg["batch_size"]
            for i in range(num_batches):
                batch_idx = range(i*cfg["batch_size"],(i+1)*cfg["batch_size"])

                x_batch = x[batch_idx,:]
                u_batch = u[batch_idx,:]
                x_next_batch = x_next[batch_idx,:]

                _, train_loss, summary_str = self.sess.run([optim, self.loss, self.summary],
                                                feed_dict={
                                                    self.x_: x_batch,
                                                    self.u_: u_batch,
                                                    self.x_next_: x_next_batch,
                                                    self.N_train : N
                                                })
                self.train_writer.add_summary(summary_str, self.counter)

                val_loss = 0.0
                if cfg["store_val"]:
                    val_loss, summary_str = self.sess.run([self.loss, self.summary],
                                                    feed_dict={
                                                        self.x_: x_val,
                                                        self.u_: u_val,
                                                        self.x_next_: x_next_val,
                                                        self.N_train : N
                                                    })
                    self.val_writer.add_summary(summary_str, self.counter)

                if (i % 10) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, train_loss: %.8f, val_loss: %.8f" \
                          % (epoch, i, num_batches, time.time() - start_time, train_loss, val_loss))
                if (self.counter % 1000) == 0:
                    print("Saving checkpoint")
                    self.saver.save(self.sess, self.writer_path + '/mlp_dyn_model', global_step=self.counter)

                self.counter += 1


    def predict(self, x_t, u_t):
        x_next_hat, log_var_x_next = self.sess.run([self.x_next_hat, self.log_var_x_next],
                                                feed_dict={
                                                    self.x_: x_t,
                                                    self.u_: u_t,
                                                    self.is_test: True
                                                })

        prediction = {
            "x": np.mean(x_next_hat, axis=1, keepdims=False),
            "aleatoric_unc": np.mean(np.exp(log_var_x_next), axis=1, keepdims=False),
            "epistemic_unc_of_mean": np.var(x_next_hat, axis=1, keepdims=False),
            "epistemic_unc_of_var": np.var(np.exp(log_var_x_next), axis=1, keepdims=False)
            }

        return prediction
