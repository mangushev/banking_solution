import os
import sys
import argparse
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from model import ExpenseEstimator
from utils import Scaler


#tf.debugging.set_log_device_placement(True)

import numpy as np
np.set_printoptions(edgeitems=25, linewidth=10000, precision=12, suppress=True)

FLAGS = None

def trans_parser(serialized_example):
  example = tf.io.parse_single_example(
    serialized_example,
    features={
      "trans": tf.io.FixedLenFeature([FLAGS.lookback_history, FLAGS.num_features], tf.float32),
      "actual": tf.io.FixedLenFeature([FLAGS.num_features], tf.float32)
    })

  return example

def balance_trans_parser(serialized_example):
  example = tf.io.parse_single_example(
    serialized_example,
    features={
      "trans": tf.io.FixedLenFeature([FLAGS.num_accounts, FLAGS.lookback_history, FLAGS.num_features], tf.float32),
      "actual": tf.io.FixedLenFeature([FLAGS.num_accounts, FLAGS.num_features], tf.float32)
    })

  return example

def metrics(actual, expense_estimate):

  performance = {}

  performance["mae"] = np.mean(np.absolute(actual - expense_estimate))
  performance["mbe"] = np.mean(actual - expense_estimate)
  #performance["rae"] = np.sum(np.absolute(actual - expense_estimate)) / np.sum(np.absolute(actual - np.mean(actual)))

  a = np.ma.masked_equal(actual,0.0)
  e = np.ma.masked_where(np.ma.getmask(a), expense_estimate)
  a = np.ma.masked_where(np.ma.getmask(a), actual)

  performance["mape"] = (np.mean(np.absolute((a - e) / a))) * 100

  performance["mse"] = np.mean(np.square(actual - expense_estimate))
  performance["rmse"] = np.sqrt(np.mean(np.square(actual - expense_estimate)))
  #performance["rse"] = np.sum(np.square(actual - expense_estimate)) / np.sum(np.square(actual - np.mean(actual)))
  performance["nrmse"] = np.sqrt(np.mean(np.square(actual - expense_estimate))) / np.std(expense_estimate)
  performance["rrmse"] = np.sqrt(np.mean(np.square(actual - expense_estimate)) / np.sum(np.square(expense_estimate)))

  return performance

def evaluate():
  #   LEGEND:
  #account / weeks / features: dimentions
  #   a - accounts
  #   t - time dimension
  #   f - period transaction totals [type1, type2, type3, ..., total_debit, total_credit]
  trans_dataset = tf.data.TFRecordDataset(FLAGS.test_file)
  trans_dataset = trans_dataset.map(trans_parser)
  trans_dataset = trans_dataset.batch(FLAGS.batch_size, drop_remainder=False)

  if FLAGS.scaler == 'Custom':
    activation_fn = tf.math.tanh
  else:
    activation_fn = tf.nn.sigmoid

  expense_estimator = ExpenseEstimator(FLAGS.batch_size, 
					FLAGS.lookback_history,
					FLAGS.num_features,
					hidden_size=FLAGS.hidden_size,
                                        activation_fn=activation_fn,
					num_hidden_layers=FLAGS.num_hidden_layers,
                     			num_attention_heads=FLAGS.num_attention_heads,
					dropout_prob=0.0,
                                        is_training=False)

  checkpoint = tf.train.Checkpoint(expense_estimator=expense_estimator)
  status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir)).expect_partial()

  expense_estimate = None
  for trans_batch in trans_dataset:
    series_data = trans_batch["trans"]
    while series_data.shape[0] != FLAGS.batch_size:
      series_data = tf.concat([series_data, tf.zeros_like(series_data[:1, :, :])], axis=0)
    estimate = expense_estimator(series_data)
    if trans_batch["trans"].numpy().shape[0] != FLAGS.batch_size:
      estimate = estimate[:trans_batch["trans"].numpy().shape[0], :]

    if expense_estimate is None:
      expense_estimate = estimate
      actual = trans_batch["actual"][:,:].numpy()
    else:
      expense_estimate = np.concatenate((expense_estimate, estimate), axis=0)
      actual = np.concatenate((actual, trans_batch["actual"][:,:].numpy()), axis=0)

  if FLAGS.scaler == 'MinMaxScaler':
    amount_scaler = joblib.load(FLAGS.scaler_file)

    expense_estimate = amount_scaler.inverse_transform(expense_estimate.reshape(-1, 1)).reshape(-1, estimate.shape[1], estimate.shape[2])
    actual = amount_scaler.inverse_transform(actual.reshape(-1, 1)).reshape(-1, trans_batch["actual"].shape[1])
  elif FLAGS.scaler == 'Custom':
    amount_scaler = Scaler(-100000, 100000, -1, 1)

    expense_estimate = amount_scaler.inverse_transform(expense_estimate.reshape(-1)).reshape(-1, estimate.shape[1], estimate.shape[2])
    actual = amount_scaler.inverse_transform(actual.reshape(-1)).reshape(-1, trans_batch["actual"].shape[1])

  debit_performance = metrics(actual[:, 0], expense_estimate[:, -1, 0])
  #credit_performance = metrics(actual[:, 1], expense_estimate[:, -1, 1])

  with tf.io.gfile.GFile(FLAGS.output_file, "w") as writer:
    writer.write("Debit" + "\n")
    for m, v in debit_performance.items():
      writer.write("{}: {:.6f} \n".format(m, v))
    #writer.write("Credit" + "\n")
    #for m, v in credit_performance.items():
    #  writer.write("{}: {:.6f} \n".format(m, v))

def predict_for_balance_maintenance():
  #weeks/account/features
  #   t - time dimension
  #   a - accounts
  #   f - period transaction totals [type1, type2, type3, ..., total_debit, total_credit]
  trans_dataset = tf.data.TFRecordDataset(FLAGS.predict_file)
  trans_dataset = trans_dataset.map(balance_trans_parser)

  if FLAGS.scaler == 'Custom':
    activation_fn = tf.math.tanh
  else:
    activation_fn = tf.nn.sigmoid

  #estimator is batched in number of accounts!
  expense_estimator = ExpenseEstimator(FLAGS.num_accounts,
					FLAGS.lookback_history,
					FLAGS.num_features,
					hidden_size=FLAGS.hidden_size,
                                        activation_fn=activation_fn,
					num_hidden_layers=FLAGS.num_hidden_layers,
                     			num_attention_heads=FLAGS.num_attention_heads,
					dropout_prob=0.0,
                                        is_training=False)

  checkpoint = tf.train.Checkpoint(expense_estimator=expense_estimator)
  status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir)).expect_partial()

  def example(estimate, actual):
    record = {
        'estimate': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(estimate, [-1]))),
        'actual': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(actual, [-1])))
    }

    return tf.train.Example(features=tf.train.Features(feature=record))

  with tf.io.TFRecordWriter(FLAGS.output_file) as writer:
    #for each week
    for trans_accounts in trans_dataset:
      trans_estimate = expense_estimator(trans_accounts["trans"])[:, -1, :]
      tf_example = example(np.array(trans_estimate, dtype=np.float32), trans_accounts["actual"][:,-2:].numpy())
      print ("estimate/actual: ", np.mean(trans_estimate.numpy(), axis=0), np.mean(trans_accounts["actual"].numpy(), axis=0))
      writer.write(tf_example.SerializeToString())

def predict():
  #   LEGEND:
  #   a - accounts
  #   t - time dimension
  #   f - period transaction totals [type1, type2, type3, ..., total_debit, total_credit]
  trans_dataset = tf.data.TFRecordDataset(FLAGS.predict_file)
  trans_dataset = trans_dataset.map(trans_parser)
  trans_dataset = trans_dataset.batch(FLAGS.batch_size, drop_remainder=False)

  if FLAGS.scaler == 'Custom':
    activation_fn = tf.math.tanh
  else:
    activation_fn = tf.nn.sigmoid

  expense_estimator = ExpenseEstimator(FLAGS.batch_size, 
					FLAGS.lookback_history,
					FLAGS.num_features,
					hidden_size=FLAGS.hidden_size,
                                        activation_fn=activation_fn,
					num_hidden_layers=FLAGS.num_hidden_layers,
                     			num_attention_heads=FLAGS.num_attention_heads,
					dropout_prob=0.0,
                                        is_training=False)

  checkpoint = tf.train.Checkpoint(expense_estimator=expense_estimator)
  status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir)).expect_partial()

  expense_estimate = None
  for trans_batch in trans_dataset:
    series_data = trans_batch["trans"]
    while series_data.shape[0] != FLAGS.batch_size:
      series_data = tf.concat([series_data, tf.zeros_like(series_data[:1, :, :])], axis=0)
    
    estimate = expense_estimator(series_data)
    if trans_batch["trans"].numpy().shape[0] != FLAGS.batch_size:
      estimate = estimate[:trans_batch["trans"].numpy().shape[0], :]

    if expense_estimate is None:
      expense_estimate = estimate
      actual = trans_batch["actual"][:,:].numpy()
    else:
      expense_estimate = np.concatenate((expense_estimate, estimate), axis=0)
      actual = np.concatenate((actual, trans_batch["actual"][:,:].numpy()), axis=0)

  if FLAGS.scaler == 'MinMaxScaler':
    amount_scaler = joblib.load(FLAGS.scaler_file)

    expense_estimate = amount_scaler.inverse_transform(expense_estimate.reshape(-1, 1)).reshape(-1, estimate.shape[1], estimate.shape[2])
    actual = amount_scaler.inverse_transform(actual.reshape(-1, 1)).reshape(-1, trans_batch["actual"].shape[1])
  elif FLAGS.scaler == 'Custom':
    amount_scaler = Scaler(-100000, 100000, -1, 1)

    expense_estimate = amount_scaler.inverse_transform(expense_estimate.reshape(-1)).reshape(-1, estimate.shape[1], estimate.shape[2])
    actual = amount_scaler.inverse_transform(actual.reshape(-1)).reshape(-1, trans_batch["actual"].shape[1])

  with tf.io.gfile.GFile(FLAGS.output_file, "w") as writer:
    writer.write("Debit: actual | estimate" + "\n")
    for a, e in zip(actual[:, 0], expense_estimate[:, -1, 0]):
      writer.write("{:.2f} | {:.2f} \n".format(a, e))
    #writer.write("Credit: actual | estimate" + "\n")
    #for a, e in zip(actual[:, 1], expense_estimate[:, -1, 1]):
    #  writer.write("{:.2f} | {:.2f} \n".format(a, e))

def train():
  #   LEGEND:
  #account / weeks / features: dimentions
  #   a - accounts
  #   t - time dimension
  #   f - period transaction totals [type1, type2, type3, ..., total_debit, total_credit]
  trans_dataset = tf.data.TFRecordDataset(FLAGS.train_file)
  trans_dataset = trans_dataset.repeat(-1)
  trans_dataset = trans_dataset.map(trans_parser)
  trans_dataset = trans_dataset.shuffle(1000, seed=0, reshuffle_each_iteration=True)
  trans_dataset = trans_dataset.batch(FLAGS.batch_size, drop_remainder=True)

  #0.00001 * 0.99 ^ (80000 / 1000) --> .0000044752
  #0.0001 * 0.99 ^ (80000 / 1000) --> .000044
  #0.0001 * 0.99 ^ (40000 / 2000) --> .00008
  #initial_learning_rate * decay_rate ^ (step / decay_steps)

  learning_rate_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.learning_rate, decay_steps=2000, decay_rate=0.99, staircase=False)

  optimizer = tf.optimizers.Adam(learning_rate_fn)

  if FLAGS.scaler == 'Custom':
    activation_fn = tf.math.tanh
  else:
    activation_fn = tf.nn.sigmoid

  expense_estimator = ExpenseEstimator(FLAGS.batch_size, 
					FLAGS.lookback_history,
					FLAGS.num_features,
					hidden_size=FLAGS.hidden_size,
                                        activation_fn=activation_fn,
					num_hidden_layers=FLAGS.num_hidden_layers,
                     			num_attention_heads=FLAGS.num_attention_heads,
					dropout_prob=FLAGS.dropout_prob,
                                        is_training=True)

  checkpoint_prefix = os.path.join(FLAGS.output_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, expense_estimator=expense_estimator)
  status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir))

  for trans_batch in trans_dataset:
    with tf.GradientTape() as tape:
      #trans_batch : [B, w, f]
      expense_estimate = expense_estimator(trans_batch["trans"])
      actual = tf.concat([trans_batch["trans"][:, 1:, :], tf.expand_dims(trans_batch["actual"], 1)], axis=1)

      #(B, w), (B, w) --> (B, w) --> (B) --> (1)
      #debit_loss = tf.reduce_mean(tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(expense_estimate[:, :, 0], actual[:, :, FLAGS.num_features-2]), axis=-1)))
      #credit_loss = tf.reduce_mean(tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(expense_estimate[:, :, 1], actual[:, :, FLAGS.num_features-1]), axis=-1)))
    
      #total_loss = debit_loss + credit_loss

      total_loss = tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(expense_estimate[:, -1, 0], trans_batch["actual"][:,0])))

      epoch = int((optimizer.iterations * FLAGS.batch_size) / FLAGS.training_set_size)

      tf.print("loss:", epoch, optimizer.iterations, total_loss, optimizer.lr(optimizer.iterations), output_stream=sys.stderr, summarize=-1)
      #tf.print("loss:", epoch, optimizer.iterations, total_loss, debit_loss, credit_loss, optimizer.lr(optimizer.iterations), output_stream=sys.stderr, summarize=-1)

    expense_estimator_gradients = tape.gradient(total_loss, expense_estimator.variables)

    optimizer.apply_gradients(zip(expense_estimator_gradients, expense_estimator.variables))

    if optimizer.iterations % FLAGS.save_batches == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    if (epoch > FLAGS.train_epochs):
      break

def main():  
  if FLAGS.action == 'TRAIN':
    train()
  elif FLAGS.action == 'EVALUATE':
    evaluate()
  elif FLAGS.action == 'PREDICT':
    predict()
  elif FLAGS.action == 'PREDICT_FOR_BALANCE':
    predict_for_balance_maintenance()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='checkpoints',
            help='Model directrory in google storage.')
    parser.add_argument('--train_file', type=str, default='data/train.tfrecords',
            help='Train file location in google storage.')
    parser.add_argument('--test_file', type=str, default='data/test.tfrecords',
            help='Test file location in google storage.')
    parser.add_argument('--predict_file', type=str, default='data/predict.tfrecords',
            help='Predict file location in google storage.')
    parser.add_argument('--output_file', type=str, default='./output.csv',
            help='Prediction output.')
    parser.add_argument('--scaler', default='Custom', choices=['MinMaxScaler','Custom'],
            help='Type of scaling used on the data.')
    parser.add_argument('--scaler_file', type=str, default='data/amount_scaler.joblib',
            help='Scaling dollar amount.')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
            help='This used for all dropouts.')
    parser.add_argument('--train_epochs', type=int, default=100,
            help='How many times to run scenarious.')
    parser.add_argument('--save_batches', type=int, default=1000,
            help='Save every N batches.')
    parser.add_argument('--num_features', type=int, default=1,
            help='How many features in traning data.')
    parser.add_argument('--num_accounts', type=int, default=4997,
            help='How many accounts.')
    parser.add_argument('--hidden_size', type=int, default=32,
            help='RNN estimator hidden size.')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
            help='One self-attention block only.')
    parser.add_argument('--num_attention_heads', type=int, default=2,
            help='number of attention heads in transformer.')
    parser.add_argument('--lookback_history', type=int, default=12,
            help='How long is sales history used by estimator.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
            help='Optimizer learning rate.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--batch_size', type=int, default=32,
            help='Batch size.')
    parser.add_argument('--training_set_size', type=int, default=422746,
            help='Batch size.')
    parser.add_argument('--action', default='PREDICT', choices=['TRAIN', 'EVALUATE', 'PREDICT', 'PREDICT_FOR_BALANCE'],
            help='An action to execure.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
