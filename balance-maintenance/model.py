
#TODO:

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_probability import distributions as tfd

import math

class Dense(tf.Module):
  def __init__(self, input_width, output_size, activation=None, stddev=1.0, name=''):
    super(Dense, self).__init__()
    self.w = tf.Variable(
      tf.random.truncated_normal([input_width, output_size], stddev=stddev), name=name + '_w')
    self.b = tf.Variable(tf.zeros([output_size]), name=name+'_b')
    self.activation = activation
    self.input_width = input_width
    self.output_size = output_size
  def __call__(self, x):
    input_shape = x.shape

    if len(input_shape) != 2 and len(input_shape) != 3:
      raise ValueError("input shape rank {} shuld be 2 or 3".format(len(input_shape)))

    if len(input_shape) == 3:
      if self.input_width != input_shape[2]:
        raise ValueError("widths do not match {} {}".format(self.input_width, input_shape[2]))
      x = tf.reshape(x, [-1, self.input_width])
    else:
      if self.input_width != input_shape[1]:
        raise ValueError("widths do not match {} {}".format(self.input_width, input_shape[1]))

    y = tf.matmul(x, self.w) + self.b
    if (self.activation is not None):
      y = self.activation(y)

    if len(input_shape) == 3:
      return tf.reshape(y, [-1, input_shape[1], self.output_size])

    return y

#Policy network
class Actor(tf.Module):
  def __init__(self, num_accounts, num_features, safety_level, transfer_limit, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
    super(Actor, self).__init__()
    self.layer1 = Dense(num_features, hidden_size, activation=None)
    self.layer2 = Dense(hidden_size, hidden_size, activation=None)
    self.layer3 = Dense(hidden_size, 2, activation=None)
    #self.mu = Dense(num_features, 1, activation=tf.math.tanh)
    #self.log_var = Dense(num_features, 1, activation=tf.nn.sigmoid)
    self.activation = activation
    self.dropout_prob = dropout_prob
    self.safety_level = safety_level
    self.transfer_limit = transfer_limit
  def __call__(self, state):
    #[I, P] --> [I]
    layer_output = self.layer1(state)
    layer_output = tfa.layers.GroupNormalization(groups = 1)(layer_output) 
    layer_output = self.activation(layer_output)
    layer_output = tf.nn.dropout(layer_output, self.dropout_prob)

    layer_output = self.layer2(layer_output)
    layer_output = tfa.layers.GroupNormalization(groups = 1)(layer_output) 
    layer_output = self.activation(layer_output)
    layer_output = tf.nn.dropout(layer_output, self.dropout_prob)

    layer_output = self.layer3(layer_output)
    #layer_output = tfa.layers.GroupNormalization(groups = 1)(layer_output) 
    #layer_output = self.activation(layer_output)
    #layer_output = tf.nn.dropout(layer_output, self.dropout_prob)

    # 0 <= u <= 1 eq 3
    #return tf.nn.relu(layer_output)
    return tf.nn.sigmoid(layer_output)*self.transfer_limit
    #return tf.math.softplus(layer_output)

  def random(self, state):
    #(a, f) --> 2*(a)
    mu_std = self(state)
    #(a)
    dist = tfd.Normal(loc=mu_std[:, 0], scale=mu_std[:, 1])
    return mu_std[:, 0], dist.entropy(), tf.squeeze(dist.sample([1]), axis=0)
          
  def best(self, state):
    #best means mean for continuous action space modeled with Normal distribution
    return self(state)[:, 0]

  def he(self, sales, x_he):
    #(1), (a), (a) --> (a)
    return tf.math.maximum(0, self.safety_level + sales - x_he)

#Value network
class Critic(tf.Module):
  def __init__(self, num_features, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
    super(Critic, self).__init__()
    self.layer1 = Dense(num_features, hidden_size, activation=None)
    self.layer2 = Dense(hidden_size, 1, activation=None)
    self.activation = activation
    self.dropout_prob = dropout_prob
  def __call__(self, state):
    #[I, P] --> [I]
    layer_output = self.layer1(state)
    layer_output = tfa.layers.GroupNormalization(groups = 1)(layer_output) 
    layer_output = self.activation(layer_output)
    layer_output = tf.nn.dropout(layer_output, self.dropout_prob)

    layer_output = self.layer2(layer_output)

    #[I, 1] --> [I]
    return tf.squeeze(layer_output, axis=-1, name='factor_squeeze')

class Env(tf.Module):
  def __init__(self, num_accounts, zero_balance, critical_balance, zero_weight, critical_weight, waste_level):
    super(Env, self).__init__()
    self.num_accounts = num_accounts
    self.zero_balance = zero_balance
    self.critical_balance = critical_balance
    self.zero_weight = zero_weight
    self.critical_weight = critical_weight
    self.waste_level = waste_level

  def reset(self, x):
    self.x = x

  def waste(self):
    return self.waste_level * self.x

  def __call__(self, u, sales):
    #(a) + (a) --> (a)
    x_u = self.x + u
    #there can be more money on the account than 100k that is used to normalize data
    #x_u = tf.math.minimum(1, self.x + u)

    overdraft = tf.math.minimum(0, x_u - sales)

    #(a) - (a) --> (a)
    self.x = tf.math.maximum(0, x_u - sales)

    z = tf.cast(self.x < self.zero_balance, tf.float32)

    critical = tf.cast(self.x < self.critical_balance, tf.float32)

    q = tf.math.maximum(0, self.waste())

    z = self.zero_weight * z

    critical = self.critical_weight * critical

    #(a), (a), (a), (a) --> (a)
    r = tf.cast(1 - z - critical - q, tf.float32)

    return self.x, overdraft, r, z, critical, q
