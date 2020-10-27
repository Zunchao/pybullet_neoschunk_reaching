import tensorflow as tf
import numpy as np
import os
import shutil
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import sys
sys.path.append(os.path.dirname(currentdir))
import datetime
from ddpg.mmRobotGymEnv import MMRobotGymEnv
from ddpg.mm2DRobotGymEnvCircle import MM2DRobotCGymEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 2000
MAX_EP_STEPS = 300
LR_A = 5e-5  # learning rate for actor
LR_C = 5e-5  # learning rate for critic
GAMMA = 0.99  # reward discount
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
MEMORY_CAPACITY = 10000
BATCH_SIZE = 256
VAR_MIN = 0.1
RENDER = 0
LOAD = 0
DISCRETE_ACTION = False



class Actor(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        self.S = tf.placeholder(tf.float32, [None, state_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, state_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(self.S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(self.S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 400, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
                # a = np.clip(np.matmul([ob], self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={self.S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = [s]  # single state
        return self.sess.run(self.a, feed_dict={self.S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        self.S = tf.placeholder(tf.float32, [None, state_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, state_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(self.S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(self.S_, a_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = self.R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]  # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 400
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 300, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={self.S: s, self.a: a, self.R: r, self.S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

def train(train_log_dir):
    var = 2.  # control exploration
    Nsuccess = 0
    Nep = 0
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_step = 0
        returns = 0
        Nep += 1
        train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        for t in range(MAX_EP_STEPS):
            # while True:
            if RENDER:
                env.render()

            # Added exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), -ACTION_BOUND, ACTION_BOUND)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                var = max([var * .9995, VAR_MIN])  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_step += 1
            returns += r
            if done or t == MAX_EP_STEPS - 1:
                tf.summary.scalar('return', returns)
                merged = tf.summary.merge_all()
                summary, returnss = sess.run([merged, returns], feed_dict=None)
                train_summary_writer.add_summary(summary,ep)
                print('Ep:', ep,
                      '| Steps: %i' % int(ep_step),
                      '| Explore: %.2f' % var,
                      '| Return: %.2f' % returns,
                      )
                break
        env.close()

        if not ep%(MAX_EPISODES/5):
            if os.path.isdir(path): shutil.rmtree(path)
            os.mkdir(path)
            ckpt_path = os.path.join(path, str(ep)+'DDPG.ckpt')
            save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
            print("\nSave Model %s\n" % save_path)


def eval(test_summary_writer):
    env.set_fps(30)
    i = 0
    while True:
        s = env.reset()
        for t in range(MAX_EP_STEPS):
            env.render()
            rec.capture_frame()
            # time.sleep(0.01)
            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)
            s = s_
            if done or t == MAX_EP_STEPS - 1:
                break

        print('EP : ', i+1,
              'Goal : ', env._goal,
              'Robot : ', env.robot_state)

        i += 1
        if i > 10:
            break
    rec.close()
    env.close()


if __name__ == '__main__':

    sess = tf.Session()
    # env = MMRobotGymEnv(maxstep=MAX_EP_STEPS, space=2)
    env = MM2DRobotCGymEnv(maxstep=MAX_EP_STEPS, space=2)
    rec = VideoRecorder(env, path='/home/zheng/fromremote/video/ddpgrollout.mp4')
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    ACTION_BOUND = env.action_bound

    # Create actor and critic.
    actor = Actor(sess, STATE_DIM, ACTION_DIM, ACTION_BOUND, LR_A, REPLACE_ITER_A)
    critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

    saver = tf.train.Saver()
    path = './discrete' if DISCRETE_ACTION else './continuous'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = path + current_time + '/train'
    test_log_dir = path + current_time + '/test'
    test_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    if LOAD:
        saver.restore(sess, tf.train.latest_checkpoint(path))
        eval(test_summary_writer)
    else:
        sess.run(tf.global_variables_initializer())
        train(train_log_dir)

