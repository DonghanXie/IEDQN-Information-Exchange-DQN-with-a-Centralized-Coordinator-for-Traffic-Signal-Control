import tensorflow as tf
import numpy as np
import os
from agents.utils import *
from agents.policies import *
import logging
import multiprocessing as mp
class IEDQN:
    def __init__(self,n_s_ls,n_a_ls,n_w_ls, total_step,
                 model_config, seed=0):
        self.name = 'iedqn'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_step= model_config.getint('batch_size')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True,gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=config)
        self.policy = self._init_policy(n_s_ls,n_a_ls,n_w_ls,self.n_agent,model_config,name=self.name)
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())
        
        
    #    
    def _init_policy(self, n_s_ls, n_a_ls, n_w_ls,n_agent,model_config,name=None):
        n_fc_info = model_config.getint('num_info')
        n_fc_mess = model_config.getint('num_mess')
        n_fc_dqn = model_config.getint('num_fc_dqn')
        dim_info = model_config.getint('dim_info')
        dim_mess = model_config.getint('dim_mess')
        
        policy = Policy(n_s_ls, n_a_ls, n_w_ls, n_fc_info , n_fc_mess , n_fc_dqn, dim_info , dim_mess , num_agent=n_agent , name=name)
        return policy
    
    def forward(self, obs,old_obs,last_out_values,mode='act'):
        if mode == 'explore':
            eps = self.eps_scheduler.get(1)
        
        cur_values = self.policy.forward(self.sess, obs, old_obs,last_out_values)
        actions = np.argmax(cur_values,axis=1)
        for i in range(self.n_agent):
            if (mode=='explore') and (np.random.random()<eps):
                actions[i] = np.random.randint(self.n_a_ls[i])
        return actions , cur_values
    
    def backward(self,summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        if self.trans_buffer.size < self.trans_buffer.batch_size:
            return
        for k in range(10):
            obs, old_obs,last_out_values,cur_values,acts, cur_rewards,next_obs, dones = self.trans_buffer.sample_transition()
            for i in range(self.n_step):
                if i == 0 :
                    self.policy.backward(self.sess,obs[i],old_obs[i],last_out_values[i],cur_values[i],acts[i],cur_rewards[i],next_obs[i], cur_lr,dones[i],
                             summary_writer=summary_writer, global_step=global_step + k)      
                else:
                    self.policy.backward(self.sess,obs[i],old_obs[i],last_out_values[i],cur_values[i],acts[i],cur_rewards[i],next_obs[i], cur_lr,dones[i])

        
        
    def _init_train(self, model_config):
        # init loss
        max_grad_norm = model_config.getfloat('max_grad_norm')
        gamma = model_config.getfloat('gamma')
        buffer_size = model_config.getfloat('buffer_size')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.policy.prepare_loss(max_grad_norm, alpha, epsilon)
        self.trans_buffer = ReplayBuffer(buffer_size, self.n_step)
            

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False
    
    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        eps_init = model_config.getfloat('epsilon_init')
        eps_decay = model_config.get('epsilon_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('LR_MIN')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if eps_decay == 'constant':
                self.eps_scheduler = Scheduler(eps_init, decay=eps_decay)
        else:
            eps_min = model_config.getfloat('epsilon_min')
            eps_ratio = model_config.getfloat('epsilon_ratio')
            self.eps_scheduler = Scheduler(eps_init, eps_min, self.total_step * eps_ratio,
                                           decay=eps_decay)    


    def save(self, model_dir, global_step):
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def add_transition(self, obs,old_obs,last_out_values,cur_values,actions,cur_rewards,next_obs,done):
        if (self.reward_norm):
            cur_rewards = cur_rewards/self.reward_norm
        if self.reward_clip:
            cur_rewards = np.clip(cur_rewards, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(obs,old_obs,last_out_values,cur_values,actions,cur_rewards,next_obs,done)

