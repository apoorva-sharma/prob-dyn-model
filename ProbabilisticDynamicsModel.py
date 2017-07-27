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


MLP_DM_cfg = {
    "lr": 2e-3,
    "beta1": 0.9,
    "batch_size": 100,
    "val_batch_size": 100,
    "n_epochs": 100,
    "store_val": True
}

# This class lays out the mechanics for training any transition dyanmics model
# that predicts a mean vector and log(variance) vector with dropout.
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
        self.counter = 0

    # Child classes must define function must set self.x_next_hat and
    # self.log_var_x_next to be tensors corresponding to the appropriate
    # calculations performed on the self.x_ and self.u_ placeholders
    def prediction_model(self):
        raise NotImplementedError("Subclass must override prediction_model()")

    def build_model(self):
        self.x_ = tf.placeholder(tf.float32, (None, self.x_dim), name="x")
        self.u_ = tf.placeholder(tf.float32, (None, self.u_dim), name="u")
        self.x_next_ = tf.placeholder(tf.float32, (None, self.x_dim), name="x_next")
        self.is_test = tf.placeholder_with_default(False, [], name="is_test")
        self.N_train = tf.placeholder(tf.float32, [], name="N_train")

        with tf.variable_scope("prob_dyn_model"):
            self.prediction_model()

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
                                                        self.x_: x_val[0:cfg["val_batch_size"],:],
                                                        self.u_: u_val[0:cfg["val_batch_size"],:],
                                                        self.x_next_: x_next_val[0:cfg["val_batch_size"],:],
                                                        self.N_train : N,
                                                        self.is_test : True
                                                    })
                    self.val_writer.add_summary(summary_str, self.counter)

                if (i % 10) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, train_loss: %.8f, val_loss: %.8f" \
                          % (epoch, i, num_batches, time.time() - start_time, train_loss, val_loss))
                if (self.counter % 1000) == 0:
                    print("Saving checkpoint")
                    self.saver.save(self.sess, self.writer_path + '/mlp_dyn_model', global_step=self.counter)

                self.counter += 1

        self.saver.save(self.sess, self.writer_path + '/mlp_dyn_model', global_step=self.counter-1)



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


class NonlinearPDM(MLPDynamicsModel):
    def prediction_model(self):
        N = tf.cond(self.is_test, lambda: tf.constant(self.num_mc_samples), lambda: tf.constant(1))
        x = add_mc_samples(self.x_, N)
        u = add_mc_samples(self.u_, N)
        model_input = tf.concat([x, u], axis=2)

        self.log_var_x_next = tf.get_variable("log_var_x_next", [1, 1, self.x_dim], tf.float32,
                              tf.constant_initializer(-5.0))
        # self.log_var_x_next = -2.0*tf.ones([1,1,self.x_dim])

        #dropout_reg = 2*tf.sqrt(tf.reduce_sum(tf.exp(self.log_var_x_next)))/self.N_train
        dropout_reg = 2*0.53/self.N_train

        # model_input has shape (K,N,M), corresponding to the K samples input
        # to the model, N = self.num_mc_samples copies of each input for MC
        # epistemic uncertainty calculation, and M is the x_dim + u_dim
        z, reg_loss = mlp_with_dropout(model_input, self.hidden_layer_sizes, self.dropout_prob, self.is_test)
        # z, reg_loss = mlp_with_concrete_dropout(model_input, self.hidden_layer_sizes, self.is_test, dropout_reg=dropout_reg)

        delta_x, reg_loss2 = dense_with_dropout(z, output_size=self.x_dim,
                                prob=self.dropout_prob, is_test=self.is_test, name="fc_mean")
        # delta_x, reg_loss2 = dense_with_concrete_dropout(z, output_size=self.x_dim,
        #                                                 is_test=self.is_test, dropout_reg=dropout_reg,
        #                                                 name="fc_mean")

        self.x_next_hat = x + delta_x

        self.reg_loss = reg_loss + reg_loss2


class LocallyLinearPDM(MLPDynamicsModel):
    def prediction_model(self):
        N = tf.cond(self.is_test, lambda: tf.constant(self.num_mc_samples), lambda: tf.constant(1))
        x = add_mc_samples(self.x_, N)
        u = add_mc_samples(self.u_, N)
        model_input = tf.concat([x, u], axis=2)

        # Variance is independent of input (homoscedastic)
        self.log_var_x_next = tf.get_variable("log_var_x_next", [1, 1, self.x_dim], tf.float32,
                                                tf.constant_initializer(-5.0))

        dropout_reg = 2*tf.sqrt(tf.reduce_sum(tf.exp(self.log_var_x_next)))/self.N_train

        # z, reg_loss = mlp_with_dropout(model_input, self.hidden_layer_sizes, self.dropout_prob, self.is_test)
        z, reg_loss = mlp_with_concrete_dropout(model_input, self.hidden_layer_sizes, self.is_test, dropout_reg=dropout_reg)

        A_dim = self.x_dim**2
        B_dim = self.x_dim*self.u_dim
        A_flat, reg_loss_A = dense_with_dropout(z, output_size=A_dim,
                               prob=self.dropout_prob, is_test=self.is_test, name="fc_A")
        B_flat, reg_loss_B = dense_with_dropout(z, output_size=B_dim,
                               prob=self.dropout_prob, is_test=self.is_test, name="fc_B")
        # A_flat, reg_loss_A = dense_with_concrete_dropout(z,
        #                        dropout_reg=dropout_reg, output_size=A_dim,
        #                        is_test=is_test, name="fc_A")
        # B_flat, reg_loss_B = dense_with_concrete_dropout(z,
        #                        dropout_reg=dropout_reg, output_size=B_dim,
        #                        is_test=is_test, name="fc_B")

        self.A = tf.reshape(A_flat, [-1, self.x_dim, self.x_dim], name="A_into_matrix")
        self.B = tf.reshape(B_flat, [-1, self.x_dim, self.u_dim], name="B_into_matrix")

        x_matrix = tf.expand_dims( tf.reshape(x, [-1, self.x_dim], name="x_into_matrix"), axis=-1)
        u_matrix = tf.expand_dims( tf.reshape(u, [-1, self.u_dim], name="u_into_matrix"), axis=-1)
        x_next_hat = x_matrix + self.A @ x_matrix + self.B @ u_matrix
        x_next_hat = tf.squeeze(x_next_hat, axis=-1)
        self.x_next_hat = tf.reshape(x_next_hat, [-1, N, self.x_dim], name="result_to_batch")


        self.reg_loss = reg_loss + reg_loss_A + reg_loss_B
