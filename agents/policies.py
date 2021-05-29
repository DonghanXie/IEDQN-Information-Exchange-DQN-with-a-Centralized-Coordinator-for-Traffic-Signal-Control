import tensorflow as tf
import numpy as np
import numpy as np
import tensorflow as tf
from agents.utils import *
import bisect


GAMMA = 0.9
class Policy:
    def __init__(self, n_s_ls, n_a_ls, n_w_ls,n_fc_info , n_fc_mess , n_fc_dqn, dim_info , dim_mess , num_agent=25,name=None):
        self.name = name
        self.n_a_ls = n_a_ls
        self.n_s_ls = n_s_ls
        self.n_w_ls = n_w_ls
        
        self.n_a = n_a_ls[0]
        self.n_s = n_s_ls[0]
        self.n_w = n_w_ls[0]
        
        self.n_fc_info = n_fc_info
        self.n_fc_mess = n_fc_mess
        self.n_fc_dqn = n_fc_dqn
        self.dim_info = dim_info
        self.dim_mess = dim_mess
        
        self.n_agent = num_agent
        
        
        self.ob_fw = tf.placeholder(tf.float32,[self.n_agent,self.n_s]) #s1   
        self.old_ob_fw = tf.placeholder(tf.float32,[self.n_agent,self.n_s]) #s0
        self.last_out_values = tf.placeholder(tf.float32,[self.n_agent,self.n_a]) #v0
        
        
        with tf.variable_scope(self.name):       
            self.v = self._build_net()
            

        

    def _build_net(self):
        ob = self.ob_fw
        old_ob = self.old_ob_fw
        last_out = self.last_out_values

        info_in = tf.concat([old_ob,last_out],1)
        info_h = fc(info_in,'info_net',self.n_fc_info)
        info_out = fc(info_h,'info_net',self.dim_info)
        
        
        mess_in = tf.reshape(info_out,[1,self.n_agent*self.dim_info])       
        mess_h = fc(mess_in,'mess_net',self.n_fc_mess)
        mess_out = fc(mess_h,'mess_net',self.dim_mess*self.n_agent)
        mess_mat = tf.reshape(mess_out,[self.n_agent,self.dim_mess])
        
        dqn_in = tf.concat([ob,mess_mat],1)
        dqn_h = fc(dqn_in,'dqn',self.n_fc_dqn)
        dqn_out = fc(dqn_h,'dqn',self.n_a, act = lambda x:x )
        
        return dqn_out


    def forward(self,sess,ob,old_ob,last_out_values):
        out_values = sess.run(self.v, {self.ob_fw:np.array(ob),
                                     self.old_ob_fw:np.array(old_ob),
                                     self.last_out_values:np.array(last_out_values)})

        return out_values
        

    def prepare_loss(self, max_grad_norm, alpha, epsilon):
        self.action_input = tf.placeholder("float",[self.n_agent,self.n_a])
        self.y_input = tf.placeholder("float",[self.n_agent])
        Q_action = tf.reduce_sum(tf.multiply(self.v,self.action_input),reduction_indices = 1)
        self.loss = tf.square(self.y_input-Q_action)
        self.loss_ = tf.reduce_mean(self.loss)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.minimize(self.loss)
 
        summaries = []
        summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss_))
        self.summary = tf.summary.merge(summaries)
            
    def backward(self, sess, cur_obs, old_obs, last_out_values, cur_values, acts, cur_rewards, next_obs,  cur_lr,done,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        y_batch = []
        values_batch = sess.run(self.v,feed_dict={self.ob_fw : next_obs, self.old_ob_fw : cur_obs , self.last_out_values: cur_values})
        if done:
            y_batch = cur_rewards
        else:
            y_batch=cur_rewards+GAMMA*np.max(values_batch,1)
        actions = np.zeros(shape=[self.n_agent,self.n_a])
        for i,j in enumerate(acts):
            actions[i,j] = 1
            
        outs = sess.run(ops,
                        {self.ob_fw : cur_obs,
                         self.old_ob_fw : old_obs,
                         self.last_out_values : last_out_values,
                         self.action_input : actions,
                         self.y_input : y_batch,
                         self.lr: cur_lr})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)
