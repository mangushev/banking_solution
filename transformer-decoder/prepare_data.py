
#TODO: 

import tensorflow as tf
import os
import argparse
import sys
import random
import math
import logging
import numpy as np
import pandas as pd
import datetime
import re
import joblib
from csv import reader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from utils import Scaler

FLAGS = None

logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.removeHandler(logger.handlers[0])
logger.propagate = False


#file 4: Transaction Date	customer_id	Amount(Debit)	FraudFlag

def create_records(trans_file1, trans_file2, trans_file3, trans_file4, train_file, test_file, balance_train_file, balance_test_file, ar_file):

  #is Sunday, add 1 
  def IntervalNumber(trans_date):
    if (FLAGS.aggregate == 'WEEK'):
      epochMonday = FLAGS.epoch_date - datetime.timedelta(days=FLAGS.epoch_date.weekday())
      todayMonday = trans_date - datetime.timedelta(days=trans_date.weekday())
      return (todayMonday - epochMonday).days / 7
    else:
      return (trans_date - FLAGS.epoch_date).days

  x = []
  with open(trans_file1, 'r') as f:
    csv_reader = reader(f, delimiter='\t')
    next(csv_reader)

    p = re.compile('^(\w+).*')

    for data in csv_reader: 
      line = []
    
      line.append(1) 
    
      trans_date = datetime.datetime.strptime(data[1], '%m/%d/%Y')
      line.append(int(IntervalNumber(trans_date)))
    
      m = re.match(p, data[2])
      #line.append(m.group(1).upper())
      line.append("Purchase")
    
      line.append(data[3])
      line.append(data[4])
    
      #x.append(line)

  with open(trans_file2, 'r') as f:
    csv_reader = reader(f, delimiter='\t')
    next(csv_reader)

    p = re.compile('^(\w+).*')
    for data in csv_reader: 
      line = []
    
      line.append(2) 
 
      trans_date = datetime.datetime.strptime(data[1], '%m/%d/%Y')
      line.append(int(IntervalNumber(trans_date)-175)) #make them start from the same week

      #make same number of weeks as the first file
      #if (int(IntervalNumber(trans_date))-25 > 2660):
      if (int(IntervalNumber(trans_date))-175 > 18622):
        continue
    
      m = re.match(p, data[2])
      #line.append(m.group(1).upper())
      line.append("Purchase")
    
      if data[4] == "Sale":
        line.append(-float(data[5]))
        line.append('')
      else:
        line.append('')      
        line.append(-float(data[5]))
    
      #x.append(line)
    
  with open(trans_file3, 'r') as f:
    csv_reader = reader(f, delimiter='\t')
    next(csv_reader)

    p = re.compile('^(\w+).*')
    for data in csv_reader: 
      line = []
    
      line.append(3) 
    
      trans_date = datetime.datetime.strptime(data[0], '%m/%d/%Y')
      #line.append(int(IntervalNumber(trans_date))-19) #make them start from the same week
      line.append(int(IntervalNumber(trans_date))-130) #make them start from the same week

      #make same number of weeks as the first file
      #if (int(IntervalNumber(trans_date))-19 > 2660):
      if (int(IntervalNumber(trans_date))-130 > 18622):
        continue
    
      m = re.match(p, data[1])
      #line.append(m.group(1).upper())
      line.append("Purchase")
    
      if data[7] == "Debit":
        line.append(-float(data[2]))
        line.append('')
      else:
        line.append('')      
        line.append(-float(data[2]))
    
      #x.append(line)

  #dict = {}
  fraud = set()
  with open(trans_file4, 'r') as f:
    csv_reader = reader(f, delimiter='\t')
    next(csv_reader)

    p = re.compile('^(\w+).*')
    for data in csv_reader: 
      line = []
    
      line.append(data[1]) 
    
      if (int(data[3]) == 1):
        fraud.add(int(data[1]))
    
      trans_date = datetime.datetime.strptime(data[0], '%m/%d/%Y')
      line.append(int(IntervalNumber(trans_date))) #make them start from the same week
      #dict[int(IntervalNumber(trans_date))] = trans_date

      line.append("Purchase")
    
      line.append(float(data[2]))
      line.append('')

      x.append(line)
      #if (len(all) > 100000):
      #  break

  x = np.asarray(x, dtype=str)

  X = x[:, 0:2]
    
  type_encoder = OrdinalEncoder()

  X = np.concatenate((X, type_encoder.fit_transform(x[:, 2:3]).astype(int) ), axis=1)

  debit = x[:, 3]
  debit = np.where(debit == '', 0, debit).astype(np.float32).reshape(-1, 1)

  credit = x[:, 4]
  credit = np.where(credit == '', 0, credit).astype(np.float32).reshape(-1, 1)

  x = None

  X = np.concatenate((X, debit), axis=1)
  X = np.concatenate((X, credit), axis=1)

  #X = np.concatenate((X, amount_scaler.transform(debit)), axis=1)
  #X = np.concatenate((X, amount_scaler.transform(credit)), axis=1)

  transactions = pd.DataFrame(X, columns = ['account_id', 'interval_number', 'transaction_type', 'debit', 'credit'])

  X = None

  transactions['account_id'] = transactions['account_id'].astype(np.int32)
  transactions['interval_number'] = transactions['interval_number'].astype(np.int32)
  transactions['transaction_type'] = transactions['transaction_type'].astype(np.int32)
  transactions['debit'] = transactions['debit'].astype(np.float32)
  transactions['credit'] = transactions['credit'].astype(np.float32)

  transactions.info(verbose=True)

  transactions_aggr = transactions.groupby(['account_id', 'interval_number', 'transaction_type'], as_index=False).agg({'debit': 'sum', 'credit': 'sum'})

  def example(trans, actual):
    record = {
        'trans': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(trans, [-1]))),
        'actual': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(actual, [-1])))
    }

    return tf.train.Example(features=tf.train.Features(feature=record))

  def ar_example(trans):
    record = {
        'trans': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(trans, [-1])))
    }

    return tf.train.Example(features=tf.train.Features(feature=record))

  accounts_list = transactions_aggr['account_id'].unique()

  starting_interval = transactions_aggr['interval_number'].min()
  ending_interval = transactions_aggr['interval_number'].max()
  transaction_types = sorted(transactions_aggr['transaction_type'].unique())

  #record format

  #- expense type 1 week sum 
  #- expense type 1 week sum
  #- expense type 1 week sum
  #...
  #- expense type 1 week sum

  #- debit total week sum
  #- credit total week sum
  #- balance

  #Future:
    
  #- salary
  #- economy
  #- weather

  accounts = []
  for a in accounts_list:
    account = []
    for w in range(starting_interval, ending_interval+1):
      interval_transactions = []
      debit = 0
      credit = 0
      for tt in transaction_types:    
        result = transactions_aggr[(transactions_aggr['account_id'] == a) & (transactions_aggr['interval_number'] == w) & (transactions_aggr['transaction_type'] == tt)][['debit', 'credit']]
        if (result.empty):
          interval_transactions.append(0)
        else:
          results_array = result.to_numpy()
          interval_transactions.append(results_array[0][0]+results_array[0][1])
          debit = debit + results_array[0][0]
          credit = credit + results_array[0][1]
#      interval_transactions.append(debit)
#      interval_transactions.append(credit)
      account.append(interval_transactions) 
    accounts.append(account)
    
  np_accounts = np.array(accounts)

  shape = np_accounts.shape

  if FLAGS.scaler == 'MinMaxScaler':
    flattened = np_accounts.reshape(shape[0], shape[1], shape[2], 1).reshape(-1, 1)

    amount_scaler = MinMaxScaler()
    amount_scaler.fit(flattened)

    joblib.dump(amount_scaler, FLAGS.scaler_file)

    scaled_accounts = amount_scaler.transform(flattened)

    scaled_accounts = scaled_accounts.reshape(shape[0], shape[1], shape[2])
  elif FLAGS.scaler == 'Custom':
    flattened = np_accounts.reshape(-1)

    amount_scaler = Scaler(-100000, 100000, -1, 1)

    scaled_accounts = amount_scaler.transform(flattened)

    scaled_accounts = scaled_accounts.reshape(shape[0], shape[1], shape[2])
  else:
    scaled_accounts = np_accounts

#  position_data = []
#  for i in range(starting_interval, ending_interval+1):
#    positions = []
#    trans_date = FLAGS.epoch_date + datetime.timedelta(days=i)
#    #print ("calc {} vs dict {}: interval {}".format(trans_date, dict[i], i))
#    positions.append(np.sin(2*np.pi*trans_date.weekday()/7))
#    positions.append(np.cos(2*np.pi*trans_date.weekday()/7))
#    positions.append(np.sin(2*np.pi*trans_date.day/30))
#    positions.append(np.cos(2*np.pi*trans_date.day/30))
#    positions.append(np.sin(2*np.pi*trans_date.month/12))
#    positions.append(np.cos(2*np.pi*trans_date.month/12))
#    position_data.append(positions)
#  
#  #history, 6
#  np_position_data = np.array(position_data)
#
  #for autoregression
  with tf.io.TFRecordWriter(ar_file) as writer:
    for a in np_accounts: #no scaling
      #a: history, features
      a_p = a #np.concatenate((a, np_position_data), axis=1)
      tf_example = ar_example(a_p[:, :])
      writer.write(tf_example.SerializeToString())

  examples = []
  for a in np.array(scaled_accounts):
    #a: history, features
    a_p = a #np.concatenate((a, np_position_data), axis=1)
    for h in range(a_p.shape[0]-FLAGS.lookback_history):
      examples.append(example(a_p[h:h+FLAGS.lookback_history, :], a_p[h+FLAGS.lookback_history, :]))
      #print (a_ph:h+FLAGS.lookback_history, :])

  random.shuffle(examples)

  train_count = 0
  with tf.io.TFRecordWriter(train_file) as writer: 
    for example in examples[:int(len(examples) * (1 - FLAGS.forecasting_testset_size))]:
      writer.write(example.SerializeToString())
      train_count = train_count + 1

  test_count = 0
  with tf.io.TFRecordWriter(test_file) as writer: 
    for example in examples[int(len(examples) * (1 - FLAGS.forecasting_testset_size)):]:
      writer.write(example.SerializeToString())
      test_count = test_count + 1

  logger.info ("fraud account: {}".format(len(fraud)))
  logger.info ("customer accounts shape: {}".format(np_accounts.shape))
  logger.info ("training/testing forecasting records: {}/{}".format(train_count, test_count))
  logger.info ("transactions types without debit, credit: {}".format(len(transaction_types)))
  logger.info ("transaction categories: {}".format(type_encoder.categories_))
  logger.info ("type numbers: {}".format(transaction_types))
  logger.info ("intervals in total - all hitory length: {}".format((ending_interval+1) - starting_interval))

  train_counter = 0
  train_test_break = int(a.shape[0] * (1 - FLAGS.balance_maintenance_testsize))
  with tf.io.TFRecordWriter(balance_train_file) as writer:
    for h in range(0, train_test_break-FLAGS.lookback_history):
      trans_history = []
      trans_actual = []
      for a in np.array(scaled_accounts):
        trans_history.append(a[h:h+FLAGS.lookback_history, :])
        trans_actual.append(a[h+FLAGS.lookback_history, :])
      tf_example = example(np.array(trans_history, dtype=np.float32), np.array(trans_actual, dtype=np.float32))
      writer.write(tf_example.SerializeToString())
      train_counter = train_counter + 1

  test_counter = 0
  with tf.io.TFRecordWriter(balance_test_file) as writer:
    for h in range(train_test_break, a.shape[0]-FLAGS.lookback_history):
      trans_history = []
      trans_actual = []
      for a in np.array(scaled_accounts):
        trans_history.append(a[h:h+FLAGS.lookback_history, :])
        trans_actual.append(a[h+FLAGS.lookback_history, :])
      tf_example = example(np.array(trans_history, dtype=np.float32), np.array(trans_actual, dtype=np.float32))
      writer.write(tf_example.SerializeToString())
      test_counter = test_counter + 1

  logger.info ("created {}/{} train/test balance maintenance records".format(train_counter, test_counter))


def main():
  create_records(FLAGS.train_file1, FLAGS.train_file2, FLAGS.train_file3, FLAGS.train_file4, FLAGS.train_tfrecords_file, FLAGS.test_tfrecords_file, FLAGS.balance_train_tfrecords_file, FLAGS.balance_test_tfrecords_file, FLAGS.ar_tfrecords_file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--train_file1', type=str, default='data/trans.txt',
            help='trans file 1.')
  parser.add_argument('--train_file2', type=str, default='data/trans2.txt',
            help='trans file 2.')
  parser.add_argument('--train_file3', type=str, default='data/text.txt',
            help='trans file 3.')
  parser.add_argument('--train_file4', type=str, default='data/simulated_without_fraud_365.txt',
            help='trans file 4.')
  parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
  parser.add_argument('--aggregate', default='WEEK', choices=['WEEK','DAY'],
            help='Agregate transactions by week or by day.')
  parser.add_argument('--train_tfrecords_file', type=str, default='data/train.tfrecords',
            help='train tfrecords output file for forecasting')
  parser.add_argument('--test_tfrecords_file', type=str, default='data/test.tfrecords',
            help='test tfrecords output file for forecasting')
  parser.add_argument('--balance_train_tfrecords_file', type=str, default='data/balance_train.tfrecords',
            help='This is to generate input for balance maintenance trainset')
  parser.add_argument('--balance_test_tfrecords_file', type=str, default='data/balance_test.tfrecords',
            help='This is to generate input for balance maintenance testset')
  parser.add_argument('--ar_tfrecords_file', type=str, default='data/sr.tfrecords',
            help='This is to test autoregression algorithm.')
  parser.add_argument('--scaler', default='None', choices=['MinMaxScaler','Custom', 'None'],
            help='Type of scaling used on the data.')
  parser.add_argument('--scaler_file', type=str, default='data/amount_scaler.joblib',
            help='Scaling dollar amount.')
  parser.add_argument('--lookback_history', type=int, default=12,
            help='How long is history used by estimator.')
  parser.add_argument('--forecasting_testset_size', type=float, default=0.1,
            help='Test set size for forecasting, the rest will be train set.')
  parser.add_argument('--balance_maintenance_testset_size', type=float, default=0.3,
            help='Test set size, the rest will be train set.')
  parser.add_argument('--epoch_date', type=datetime.datetime.fromisoformat, default='1970-01-06',
            help='The date to calculate weeks from.')

  FLAGS, unparsed = parser.parse_known_args()

  logger.setLevel(FLAGS.logging)

  logger.debug ("Running with parameters: {}".format(FLAGS))

  main()
