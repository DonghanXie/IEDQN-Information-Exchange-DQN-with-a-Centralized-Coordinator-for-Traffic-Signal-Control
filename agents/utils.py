import numpy as np
import random
import tensorflow as tf

"""
layers
"""
def fc(x, scope, n_out, act=tf.nn.relu):
    with tf.variable_scope(scope):
        n_in = x.shape[-1].value
        w = tf.Variable(tf.random_normal([n_in,n_out],stddev=0.1,mean=0))
        b = tf.Variable(tf.random_normal([n_out],stddev=0.1,mean=0))
        z = tf.matmul(x, w) + b
        return act(z)

"""
buffers
"""
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma):
        self.gamma = gamma
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.dones = [done]

    def add_transition(self, ob, a, r, v, done):
        self.obs.append(ob)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()#为什么要reverse
        self.Rs = Rs
        self.Advs = Advs

    def sample_transition(self, R, discrete=True):
        self._add_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        if discrete:
            acts = np.array(self.acts, dtype=np.int32)
        else:
            acts = np.array(self.acts, dtype=np.float32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, acts, dones, Rs, Advs


class ReplayBuffer(TransBuffer):
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.cum_size = 0
        self.buffer = []
        #self.trans_buffer.add_transition(obs,old_obs,last_out_values,actions,cur_rewards,next_obs,done)

    def add_transition(self, obs,old_obs,last_out_values,cur_values,actions,cur_rewards,next_obs,done):
        experience = (obs,old_obs,last_out_values,cur_values,actions,cur_rewards,next_obs,done)
        if self.cum_size < self.buffer_size:
            self.buffer.append(experience)
        else:
            ind = int(self.cum_size % self.buffer_size)
            self.buffer[ind] = experience
        self.cum_size += 1

    def reset(self):
        self.buffer = []
        self.cum_size = 0

    def sample_transition(self):
        # Randomly sample batch_size examples
        minibatch = random.sample(self.buffer, self.batch_size)
        obs_batch = np.asarray([data[0] for data in minibatch])
        old_obs_batch = np.asarray([data[1] for data in minibatch])
        last_out_values_batch = np.asarray([data[2] for data in minibatch])
        cur_values_batch = np.asarray([data[3] for data in minibatch])     
        actions_batch = np.asarray([data[4] for data in minibatch])
        cur_rewards_batch = np.asarray([data[5] for data in minibatch])
        next_obs_batch = np.asarray([data[6] for data in minibatch])
        done_batch = np.asarray([data[7] for data in minibatch])
        return obs_batch, old_obs_batch,last_out_values_batch,cur_values_batch,actions_batch,cur_rewards_batch,next_obs_batch,done_batch

    @property
    def size(self):
        return min(self.buffer_size, self.cum_size)

"""
util functions
"""
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val


if __name__ == '__main__':
    test_layers()
