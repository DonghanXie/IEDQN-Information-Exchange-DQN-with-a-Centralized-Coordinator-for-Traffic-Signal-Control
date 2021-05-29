"""
Main function for training and evaluating agents in traffic envs
@author: Donghan Xie
"""

import argparse
import configparser
import logging
import tensorflow as tf
import threading
# from envs.test_env import GymEnv
from envs.large_grid_env import LargeGridEnv, LargeGridController
from agents.models import IEDQN
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag,
                   plot_evaluation, plot_train)



def parse_args():
    default_base_dir = './iedqn'
    default_config_dir = './config/config_large.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--port', type=int, required=False,
                        default=0)
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--test-mode', type=str, required=False,
                    default='no_test',
                    help="test mode during training",
                    choices=['no_test', 'in_train_test', 'after_train_test', 'all_test'])
    sp.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp.add_argument('--restore', type=bool, required=False,
                    default=False)
    sp = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp.add_argument('--agents', type=str, required=False,
                    default='naive', help="agent folder names for evaluation, split by ,")
    sp.add_argument('--evaluate-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(10000, 100001, 10000)]),
                    help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def train(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    in_test, post_test = init_test_flag(args.test_mode)
    port = args.port
    # init env
    env = LargeGridEnv(config['ENV_CONFIG'], port=port)
    logging.info('Training: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r, wait dim ls:%r, wave-dim-ls :%r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls,env.n_w_ls,list(map(lambda x: x[0]-x[1]-x[2],zip(env.n_s_ls,env.n_f_ls,env.n_w_ls)))))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    save_step = int(config.getfloat('TRAIN_CONFIG', 'save_step'))    
    global_counter = Counter(total_step, test_step, log_step,save_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    # coord = tf.train.Coordinator()

    model = IEDQN(env.n_s_ls, env.n_a_ls, env.n_w_ls, total_step, config['MODEL_CONFIG'],
                    seed=seed)

    if args.restore:
        model.load(base_dir+'/model/')
        
    summary_writer = tf.summary.FileWriter(dirs['log'])
    trainer = Trainer(env, model, global_counter, summary_writer, in_test, output_path=dirs['data'],save_path=dirs['model'])
    try:
        trainer.run()
    except:
        trainer.save_data()
        cur_step = global_counter.cur_step
        model.save(dirs['model'],cur_step)


    if post_test:
        tester = Tester(env, model, global_counter, summary_writer, dirs['data'])
        tester.run_offline(dirs['data'])

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model'], final_step)


def evaluate_fn(agent_dir, output_dir, seeds, port):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file for env
    config_dir = find_file(agent_dir)
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = LargeGridEnv(config['ENV_CONFIG'], port=port)
    logging.info('Evaluation: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))
    env.init_test_seeds(seeds)

    # load model for agent
    model = IEDQN(env.n_s_ls,env.n_a_ls,env.n_w_ls,0,config['MODEL_CONFIG'],seed=0)
    model.load(agent_dir + '/')
    env.agent = agent
    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
    init_log(dirs['eva_log'])
    agents = args.agents.split(',')
    # enforce the same evaluation seeds across agents
    seeds = args.evaluate_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    threads = []
    for i, agent in enumerate(agents):
        agent_dir = base_dir + '/' + agent
        thread = threading.Thread(target=evaluate_fn,
                                  args=(agent_dir, dirs['eva_data'], seeds, i))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
