import os
import sys
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from model import Actor, Critic, Env

#tf.debugging.set_log_device_placement(True)

import numpy as np
np.set_printoptions(edgeitems=25, linewidth=10000, precision=12, suppress=True)

FLAGS = None

def trans_parser(serialized_example):
  example = tf.io.parse_single_example(
    serialized_example,
    features={ 
      "estimate": tf.io.FixedLenFeature([FLAGS.num_accounts, 1], tf.float32),
      "actual": tf.io.FixedLenFeature([FLAGS.num_accounts, 1], tf.float32)
    })

  return example

def predict():
  trans_dataset = tf.data.TFRecordDataset(FLAGS.predict_file)
  trans_dataset = trans_dataset.map(trans_parser)
  trans_dataset = trans_dataset.batch(FLAGS.batch_size, drop_remainder=True)

  actor = Actor(FLAGS.num_accounts, FLAGS.num_features, FLAGS.safety_level, FLAGS.transfer_limit, FLAGS.hidden_size, activation=tf.nn.relu, dropout_prob=FLAGS.dropout_prob)

  #0.1 is 10k when date normalized to +-100k min/max
  x = tf.random.uniform(shape=[FLAGS.num_accounts], minval=0.05, maxval=0.15, dtype=tf.dtypes.float32)
  x_he = x

  env = Env(FLAGS.num_accounts, FLAGS.zero_balance, FLAGS.critical_balance, FLAGS.zero_weight, FLAGS.critical_weight, FLAGS.waste)
  env_he = Env(FLAGS.num_accounts, FLAGS.zero_balance, FLAGS.critical_balance, FLAGS.zero_weight, FLAGS.critical_weight, FLAGS.waste)

  env.reset(x)
  env_he.reset(x)

  checkpoint = tf.train.Checkpoint(actor=actor)
  checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir)).expect_partial()

  with tf.io.gfile.GFile(FLAGS.output_file, "w") as writer:
    for batch in trans_dataset:
      for i in range(FLAGS.batch_size):
        debit_estimate = batch["estimate"][i, :, 0]
        debit = batch["actual"][i, : ,0]

        q_estimate = x*FLAGS.waste
        if (FLAGS.use_actual):
          s = tf.transpose(tf.stack([x, debit, q_estimate], axis=0), perm=[1, 0])
        else:
          s = tf.transpose(tf.stack([x, debit_estimate, q_estimate], axis=0), perm=[1, 0])
        u = actor.best(s)

        x_prime, overdraft, r, z, critical, q = env(u, debit)

        u_he = actor.he(debit, x_he)
        x_prime_he, overdraft_he, r_he, z_he, critical_he, q_he = env_he(u_he, debit)

        writer.write("stock:" + ','.join(  list(map(str,   x.numpy()    ))    ) + "\n")
        writer.write("action:" + ','.join(  list(map(str,   u.numpy()    ))    ) + "\n")
        writer.write("overdraft:" + ','.join(  list(map(str,   overdraft.numpy()    ))    ) + "\n")

        writer.write("stock_he:" + ','.join(  list(map(str,   x_he.numpy()    ))    ) + "\n")
        writer.write("action_he:" + ','.join(  list(map(str,   u_he.numpy()    ))    ) + "\n")
        writer.write("overdraft_he:" + ','.join(  list(map(str,   overdraft_he.numpy()    ))    ) + "\n")

        writer.write("estimate:" + ','.join(  list(map(str,   debit_estimate.numpy()    ))    ) + "\n")
        writer.write("debit:" + ','.join(  list(map(str,   debit.numpy()    ))    ) + "\n")

        x = x_prime
        x_he = x_prime_he

def train():
  #   Dimentions LEGEND:
  #   a - accounts
  #   f - state features
  #   t - timesteps

  trans_dataset = tf.data.TFRecordDataset(FLAGS.train_file).window(FLAGS.batch_size, shift=FLAGS.batch_size-1, drop_remainder=True)

  actor_learning_rate_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.actor_learning_rate, decay_steps=1000, decay_rate=0.99, staircase=False)
  critic_learning_rate_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.critic_learning_rate, decay_steps=1000, decay_rate=0.99, staircase=False)

  actor_optimizer = tf.optimizers.Adam(actor_learning_rate_fn)
  critic_optimizer = tf.optimizers.Adam(critic_learning_rate_fn)

  #Policy and Value networks with random weights 
  actor = Actor(FLAGS.num_accounts, FLAGS.num_features, FLAGS.safety_level, FLAGS.transfer_limit, FLAGS.hidden_size, activation=tf.nn.relu, dropout_prob=FLAGS.dropout_prob)
  critic = Critic(FLAGS.num_features, FLAGS.hidden_size, activation=tf.nn.relu, dropout_prob=FLAGS.dropout_prob)
  env = Env(FLAGS.num_accounts, FLAGS.zero_balance, FLAGS.critical_balance, FLAGS.zero_weight, FLAGS.critical_weight, FLAGS.waste)

  checkpoint_prefix = os.path.join(FLAGS.output_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(critic_optimizer=critic_optimizer, actor_optimizer=actor_optimizer, critic=critic, actor=actor)
  status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir))

  episode = -1
  while (episode < FLAGS.train_episodes):
    #random initial inventory, 5k - 15k balances
    x = tf.random.uniform(shape=[FLAGS.num_accounts], minval=0.05, maxval=0.15, dtype=tf.dtypes.float32)
    env.reset(x)
    q_estimate = x * FLAGS.waste

    count = 0
    for batch_dataset in trans_dataset:
      with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        experience_step = tf.constant(0)
        experience_s = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts, FLAGS.num_features]), name="experience_s")
        experience_p = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts]), name="experience_p")
        experience_sample = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts]), name="experience_sample")
        experience_entropy = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts]), name="experience_sample")
        experience_s_prime = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts, FLAGS.num_features]), name="experience_s_prime")
        experience_r = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts]), name="experience_r_prime")
        experience_z = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts]), name="experience_z")
        experience_critical = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts]), name="experience_critical")
        experience_q = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts]), name="experience_q")
        experience_debit = tf.TensorArray(size=FLAGS.batch_size, dtype=tf.float32, element_shape=tf.TensorShape([FLAGS.num_accounts]), name="experience_debit")

        batch_iterator = batch_dataset.map(trans_parser)

        debit_estimate = next(iter(batch_iterator))['estimate'][:, 0]
        debit = next(iter(batch_iterator))['actual'][:, 0]

        if (FLAGS.use_actual):
          s = tf.transpose(tf.stack([x, debit, q_estimate], axis=0), perm=[1, 0])
        else:
          s = tf.transpose(tf.stack([x, debit_estimate, q_estimate], axis=0), perm=[1, 0])

        for item in batch_iterator:
          policy_probs, dist_entropy, policy_sampled = actor.random(s)

          experience_s = experience_s.write(experience_step, s)

          x, overdraft, r, z, critical, q = env(policy_sampled, debit)

          debit_estimate = item['estimate'][:, 0]
          debit = item['actual'][:, 0]

          #environment uses actual debit to calculate balance and waste. actor / critic are using estimates of debit and waste
          q_estimate = x * FLAGS.waste

          # if this is the last item in a batch, s will carry over to the next batch since batch shift is batch_size-1
          #(a), (a), (a) --> (f, a) --> (a, f)
          if (FLAGS.use_actual):
            s = tf.transpose(tf.stack([x, debit, q_estimate], axis=0), perm=[1, 0])
          else:
            s = tf.transpose(tf.stack([x, debit_estimate, q_estimate], axis=0), perm=[1, 0])

          experience_p = experience_p.write(experience_step, policy_probs)
          experience_sample = experience_sample.write(experience_step, policy_sampled)
          experience_entropy = experience_entropy.write(experience_step, dist_entropy)
          experience_s_prime = experience_s_prime.write(experience_step, s)
          experience_r = experience_r.write(experience_step, r)
          experience_z = experience_z.write(experience_step, z)
          experience_critical = experience_critical.write(experience_step, critical)
          experience_q = experience_q.write(experience_step, q_estimate)
          experience_debit = experience_debit.write(experience_step, debit)

          experience_step = experience_step + 1

        #(t, a, f) --> (t*a, f)
        s_batch = tf.reshape(experience_s.stack()[:experience_step, :, :], [-1, FLAGS.num_features])
        x_batch = tf.reshape(experience_s.stack()[:experience_step, :, 0], [-1])
        sal_bat = tf.reshape(experience_s.stack()[:experience_step, :, 1], [-1])
        p_batch = tf.reshape(experience_p.stack()[:experience_step, :], [-1])
        sample_batch = tf.reshape(experience_sample.stack()[:experience_step, :], [-1])
        entropy_batch = tf.reshape(experience_entropy.stack()[:experience_step, :], [-1])
        s_prime_batch = tf.reshape(experience_s_prime.stack()[:experience_step, :, :], [-1, FLAGS.num_features])
        r_batch = tf.reshape(experience_r.stack()[:experience_step, :], [-1])
        z_batch = tf.reshape(experience_z.stack()[:experience_step, :], [-1])
        critical_batch = tf.reshape(experience_critical.stack()[:experience_step, :], [-1])
        q_batch = tf.reshape(experience_q.stack()[:experience_step, :], [-1])
        debit_batch = tf.reshape(experience_debit.stack()[:experience_step, :], [-1])

        tf.print("rewards:", actor_optimizer.iterations, episode, tf.reduce_mean(r_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("overdrafts:", actor_optimizer.iterations, episode, tf.reduce_mean(z_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("critical:", actor_optimizer.iterations, episode, tf.reduce_mean(critical_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("waste:", actor_optimizer.iterations, episode, tf.reduce_mean(q_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)

        tf.print("x    :", actor_optimizer.iterations, episode, tf.reduce_mean(x_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("p    :", actor_optimizer.iterations, episode, tf.reduce_mean(p_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("sample:", actor_optimizer.iterations, episode, tf.reduce_mean(sample_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("estimate:", actor_optimizer.iterations, episode, tf.reduce_mean(sal_bat, keepdims=False), output_stream=sys.stderr, summarize=-1)
        tf.print("debit:", actor_optimizer.iterations, episode, tf.reduce_mean(debit_batch, keepdims=False), output_stream=sys.stderr, summarize=-1)

        #(t*a, f) --> (t*a)
        v = critic(s_batch)

        #(t*a, f) --> (t*a)
        v_prime = critic(s_prime_batch)

        y = r_batch + FLAGS.gamma*v_prime

        #(t*a, t*a, t*a) --> (t*a)
        delta = y - v
        tf.print("delta:", actor_optimizer.iterations, episode, tf.reduce_mean(delta, keepdims=False), output_stream=sys.stderr, summarize=-1)

        #(t*a) --> (1)
        critic_loss = 0.5*tf.reduce_mean(tf.math.square(delta), keepdims=False)
        tf.print("critic loss:", actor_optimizer.iterations, episode, critic_loss, output_stream=sys.stderr, summarize=-1)

        if actor_optimizer.iterations == 0: #for PPO
          tf.print("p_old == p_batch:", output_stream=sys.stderr, summarize=-1)
          sample_old = sample_batch #sampled policy

        #(t*a, n), (t*a, n) --> (t*a) --> (1)
        entropy_p = -tf.reduce_mean(entropy_batch) #average entropy of sample's Normal distributions
        tf.print("entropy adjusted:", actor_optimizer.iterations, episode, FLAGS.entropy_coefficient*entropy_p, output_stream=sys.stderr, summarize=-1)

        delta_stopped = tf.stop_gradient(delta)

        if FLAGS.algorithm == 'A2C':
          #(t*a), (t*a), (1) --> (1), (1) --> (1)
          #possible: substract entropy from each and mean
          #actor_loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(1e-15, sample_batch))*delta_stopped, keepdims=False) #- FLAGS.entropy_coefficient*entropy_p
          actor_loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(1e-15, sample_batch))*delta_stopped, keepdims=False) - FLAGS.entropy_coefficient*entropy_p

        elif FLAGS.algorithm == 'PPO':
          r = sample_batch/sample_old

          #(t*a,), (t*a) --> (1)
          actor_loss = -tf.reduce_mean(tf.math.minimum(r*delta_stopped,tf.clip_by_value(r,1-0.2,1+0.2)*delta_stopped), keepdims=False) - FLAGS.entropy_coefficient*entropy_p

        tf.print("actor loss:", actor_optimizer.iterations, episode, actor_loss, output_stream=sys.stderr, summarize=-1)

        sample_old = sample_batch

      actor_gradients = actor_tape.gradient(actor_loss, actor.variables)
      critic_gradients = critic_tape.gradient(critic_loss, critic.variables)
      count = count + 1

      actor_optimizer.apply_gradients(zip(actor_gradients, actor.variables))
      critic_optimizer.apply_gradients(zip(critic_gradients, critic.variables))

    episode = int(actor_optimizer.iterations / count)

    if (episode + 1) % FLAGS.save_episodes == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

def main():  
  if FLAGS.action == 'TRAIN':
    train()
  elif FLAGS.action == 'PREDICT':
    predict()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='checkpoints',
            help='Model directrory in google storage.')
    parser.add_argument('--train_file', type=str, default='data/train.tfrecords',
            help='Train file location in google storage.')
    parser.add_argument('--predict_file', type=str, default='data/test.tfrecords',
            help='Predict/Test file location in google storage.')
    parser.add_argument('--output_file', type=str, default='./output.csv',
            help='Prediction output.')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
            help='This used for all dropouts.')
    parser.add_argument('--train_episodes', type=int, default=600,
            help='How many times to run scenarious.')
    parser.add_argument('--save_episodes', type=int, default=10,
            help='Save every N episodes.')
    parser.add_argument('--num_accounts', type=int, default=4997,
            help='How many customer accounts.')
    parser.add_argument('--num_features', type=int, default=3,
            help='How many features in Critic/Actor network.')
    parser.add_argument('--hidden_size', type=int, default=96,
            help='Actor and Critic layers hidden size.')
    parser.add_argument('--entropy_coefficient', type=float, default=0.0001,
            help='Applied to entropy regularizing value for actor loss.')
    parser.add_argument('--gamma', type=float, default=0.99,
            help='Discount in future rewards.')
    parser.add_argument('--algorithm', default='A2C', choices=['A2C','PPO'],
            help='Learning algorithm for critic and actor.')
    parser.add_argument('--critical_balance', type=float, default=0.005,
            help='Critical balance comes earlier than overdraft as an additional warning to the model.')
    parser.add_argument('--zero_weight', type=float, default=0.1,
            help='Coefficient applied to number of overdrafts in reward formula.')
    parser.add_argument('--critical_weight', type=float, default=0.1,
            help='Coefficient applied to number of critical balance breach in reward formula.')
    parser.add_argument('--waste', type=float, default=32,
            help='Waste is an opportunity loss for customer. It signals to the model to avoid piling monet to the account.')
    parser.add_argument('--safety_level', type=float, default=0.05,
            help='Balance safety level for heuristic computation.')
    parser.add_argument('--transfer_limit', type=float, default=0.1,
            help='Maximum action - transfer amount. 0.1 is 10k.')
    parser.add_argument('--use_actual', default=False, action='store_true',
            help='Use debit estimates values or use actual (crystal ball) values.')
    parser.add_argument('--actor_learning_rate', type=float, default=0.001,
            help='Optimizer learning rate for Actor.')
    parser.add_argument('--critic_learning_rate', type=float, default=0.001,
            help='Optimizer learning rate for Critic.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--zero_balance', type=float, default=1e-5,
            help='Consider as zero balance if less than that.')
    parser.add_argument('--batch_size', type=int, default=2,
            help='Batch size.')
    parser.add_argument('--action', default='PREDICT', choices=['TRAIN','PREDICT'],
            help='An action to execure.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
