import logging
import numpy as np
import pandas as pd
import subprocess
import os
import sys
sys.path.append(os.path.join(os.environ['SUMO_HOME'],'tools'))
from sumolib import checkBinary
import time
import traci
import xml.etree.cElementTree as ET

DEFAULT_PORT = 8000
SEC_IN_MS = 1000

# hard code real-net reward norm
REALNET_REWARD_NORM = 20

class PhaseSet:
    def __init__(self, phases):
        self.num_phase = len(phases)
        self.num_lane = len(phases[0])
        self.phases = phases
        # self._init_phase_set()

    @staticmethod
    def _get_phase_lanes(phase, signal='r'):
        phase_lanes = []
        for i, l in enumerate(phase):
            if l == signal:
                phase_lanes.append(i)
        return phase_lanes

    def _init_phase_set(self):
        self.red_lanes = []
        # self.green_lanes = []
        for phase in self.phases:
            self.red_lanes.append(self._get_phase_lanes(phase))
            # self.green_lanes.append(self._get_phase_lanes(phase, signal='G'))


class PhaseMap:
    def __init__(self):
        self.phases = {}

    def get_phase(self, phase_id, action):
        # phase_type is either green or yellow
        return self.phases[phase_id].phases[int(action)]

    def get_phase_num(self, phase_id):
        return self.phases[phase_id].num_phase

    def get_lane_num(self, phase_id):
        # the lane number is link number
        return self.phases[phase_id].num_lane

    def get_red_lanes(self, phase_id, action):
        # the lane number is link number
        return self.phases[phase_id].red_lanes[int(action)]


class Node:
    def __init__(self, name, control=False):
        self.control = control # disabled
        self.lanes_in = []
        self.ilds_in = [] # for state
        self.name = name
        self.num_state = 0 # wave and wait should have the same dim
        self.wave_state = [] # local state
        self.wait_state = [] # local state
        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1


class TrafficSimulator:
    def __init__(self, config, output_path, is_record, record_stats, port=0):
        self.name = config.get('scenario')
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)  
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.data_path = config.get('data_path')
        self.agent = config.get('agent')
        self.cur_episode = 0
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait')}
        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait')}
        self.coef_wait = config.getfloat('coef_wait')
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]
        self._init_map()
        self.init_data(is_record, record_stats, output_path)
        self.init_test_seeds(test_seeds)
        self._init_sim(self.seed)
        self._init_nodes()
        self.terminate()

    def _debug_traffic_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase = self.sim.trafficlight.getRedYellowGreenState(self.node_names[0])
            cur_traffic = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'node': node_name,
                           'action': node.prev_action,
                           'phase': phase}
            for i, ild in enumerate(node.ilds_in):
                cur_name = 'lane%d_' % i
                lane_name = 'e:' + ild.split(':')[1]
                cur_traffic[cur_name + 'queue'] = self.sim.lane.getLastStepHaltingNumber(lane_name)
                cur_traffic[cur_name + 'flow'] = self.sim.lane.getLastStepVehicleNumber(lane_name)
                # cur_traffic[cur_name + 'wait'] = node.waits[i]
            self.traffic_data.append(cur_traffic)

    def _get_node_phase(self, action, node_name, phase_type):
        node = self.nodes[node_name]
        cur_phase = self.phase_map.get_phase(node.phase_id, action)
        if phase_type == 'green':
            return cur_phase
        prev_action = node.prev_action
        node.prev_action = action
        if (prev_action < 0) or (action == prev_action):
            return cur_phase
        prev_phase = self.phase_map.get_phase(node.phase_id, prev_action)
        switch_reds = []
        switch_greens = []
        for i, (p0, p1) in enumerate(zip(prev_phase, cur_phase)):
            if (p0 in 'Gg') and (p1 == 'r'):
                switch_reds.append(i)
            elif (p0 in 'r') and (p1 in 'Gg'):
                switch_greens.append(i)
        if not len(switch_reds):
            return cur_phase
        yellow_phase = list(cur_phase)
        for i in switch_reds:
            yellow_phase[i] = 'y'
        for i in switch_greens:
            yellow_phase[i] = 'r'
        return ''.join(yellow_phase)

    def _get_node_phase_id(self, node_name):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_node_state_num(self, node):
        assert len(node.lanes_in) == self.phase_map.get_lane_num(node.phase_id)
        # wait / wave states for each lane
        return len(node.ilds_in)

    def _get_state(self):
        # hard code the state ordering as wave, wait, fp
        state = []
        # measure the most recent state
        self._measure_state_step()

        # get the appropriate state vectors
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # wave is required in state
            cur_state = [node.wave_state]
            # include wait state
            if 'wait' in self.state_names:
                cur_state.append(node.wait_state)
            state.append(np.concatenate(cur_state))

        return state

    def _init_nodes(self):
        nodes = {}
        for node_name in self.sim.trafficlight.getIDList():
            nodes[node_name] = Node(node_name,
                                    control=True)
            lanes_in = self.sim.trafficlight.getControlledLanes(node_name)
            nodes[node_name].lanes_in = lanes_in

            ilds_in = []
            for lane_name in lanes_in:
                ild_name = 'ild:' + lane_name.split(':')[-1]
                if ild_name not in ilds_in:
                    ilds_in.append(ild_name)
            nodes[node_name].ilds_in = ilds_in
        self.nodes = nodes
        self.node_names = sorted(list(nodes.keys()))
        s = 'Env: init %d node information:\n' % len(self.node_names)
        for node in self.nodes.values():
            s += node.name + ':\n'
            s += '\tilds_in: %r\n' % node.ilds_in
        logging.info(s)
        self._init_action_space()
        self._init_state_space()

    def _init_action_space(self):
        self.n_a_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase_id = self._get_node_phase_id(node_name)
            node.phase_id = phase_id
            node.n_a = self.phase_map.get_phase_num(phase_id)
            self.n_a_ls.append(node.n_a)
        self.n_a = np.prod(np.array(self.n_a_ls))

    def _init_map(self):
        self.phase_map = None
        self.state_names = None
        raise NotImplementedError()

    def _init_sim(self, seed, gui=False):
        for i in range(100):
            try:
                sumocfg_file = self._init_sim_config(seed)
                if gui:
                    app = 'sumo-gui'
                else:
                    app = 'sumo'
                command = [checkBinary(app), '-c', sumocfg_file]
                command += ['--seed', str(seed)]
                command += ['--remote-port', str(self.port+i)]
                command += ['--no-step-log', 'True']
                if self.name != 'real_net':
                    command += ['--time-to-teleport', '600'] # long teleport for safety
                else:
                    command += ['--time-to-teleport', '300']
                command += ['--no-warnings', 'True']
                command += ['--duration-log.disable', 'True']
                # collect trip info if necessary
                if self.is_record:
                    command += ['--tripinfo-output',
                                self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
                subprocess.Popen(command)
                # wait 2s to establish the traci server
                time.sleep(2)
                self.sim = traci.connect(port=self.port+i)
                break
            except:
                continue

    def _init_sim_config(self):
        # needs to be overwriteen
        raise NotImplementedError()

    def _init_sim_traffic(self):
        return

    def _init_state_space(self):
        self._reset_state()
        self.n_s_ls = []
        self.n_w_ls = []
        self.n_f_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            num_wave = node.num_state
            num_wait = 0 if 'wait' not in self.state_names else node.num_state
            self.n_s_ls.append(num_wave + num_wait) 
            self.n_w_ls.append(num_wait)
        self.n_s = np.sum(np.array(self.n_s_ls))

    def _measure_reward_step(self):
        rewards = []
        for node_name in self.node_names:
            queues = []
            waits = []
            for ild in self.nodes[node_name].ilds_in:
                if self.obj in ['queue', 'hybrid']:
                    cur_queue = self.sim.lanearea.getLastStepHaltingNumber(ild)
                    queues.append(cur_queue)
                if self.obj in ['wait', 'hybrid']:
                    max_pos = 0
                    car_wait = 0
                    cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
                    for vid in cur_cars:
                        car_pos = self.sim.vehicle.getLanePosition(vid)
                        if car_pos > max_pos:
                            max_pos = car_pos
                            car_wait = self.sim.vehicle.getWaitingTime(vid)
                    waits.append(car_wait)
            queue = np.sum(np.array(queues)) if len(queues) else 0
            wait = np.sum(np.array(waits)) if len(waits) else 0
            if self.obj == 'queue':
                reward = - queue
            elif self.obj == 'wait':
                reward = - wait
            else:
                reward = - queue - self.coef_wait * wait
            rewards.append(reward)
        return np.array(rewards)

    def _measure_state_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            for state_name in self.state_names:
                if state_name == 'wave':
                    cur_state = []
                    for ild in node.ilds_in:
                        cur_wave = self.sim.lanearea.getLastStepVehicleNumber(ild)
                        cur_state.append(cur_wave)
                    cur_state = np.array(cur_state)
                else:
                    cur_state = []
                    for ild in node.ilds_in:
                        max_pos = 0
                        car_wait = 0
                        cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
                        for vid in cur_cars:
                            car_pos = self.sim.vehicle.getLanePosition(vid)
                            if car_pos > max_pos:
                                max_pos = car_pos
                                car_wait = self.sim.vehicle.getWaitingTime(vid)
                        cur_state.append(car_wait)
                    cur_state = np.array(cur_state)
                if self.record_stats:
                    self.state_stat[state_name] += list(cur_state)
                # normalization
                norm_cur_state = self._norm_clip_state(cur_state,
                                                       self.norms[state_name],
                                                       self.clips[state_name])
                if state_name == 'wave':
                    node.wave_state = norm_cur_state
                else:
                    node.wait_state = norm_cur_state

    def _measure_traffic_step(self):
        cars = self.sim.vehicle.getIDList()
        num_tot_car = len(cars)
        num_in_car = self.sim.simulation.getDepartedNumber()
        num_out_car = self.sim.simulation.getArrivedNumber()
        if num_tot_car > 0:
            avg_waiting_time = np.mean([self.sim.vehicle.getWaitingTime(car) for car in cars])
            avg_speed = np.mean([self.sim.vehicle.getSpeed(car) for car in cars])
        else:
            avg_speed = 0
            avg_waiting_time = 0
        # all trip-related measurements are not supported by traci,
        # need to read from outputfile afterwards
        queues = []
        for node_name in self.node_names:
            for ild in self.nodes[node_name].ilds_in:
                if self.name == 'real_net':
                    # lane_name = ild.split(':')[1]
                    lane_name = ild
                else:
                    lane_name = 'e:' + ild.split(':')[1]
                queues.append(self.sim.lane.getLastStepHaltingNumber(lane_name))
        avg_queue = np.mean(np.array(queues))
        std_queue = np.std(np.array(queues))
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.cur_sec,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'std_queue': std_queue,
                       'avg_queue': avg_queue}
        self.traffic_data.append(cur_traffic)

    @staticmethod
    def _norm_clip_state(x, norm, clip=-1):
        x = x / norm
        return x if clip < 0 else np.clip(x, 0, clip)

    def _reset_state(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # prev action for yellow phase before each switch
            node.prev_action = 0
            node.num_state = self._get_node_state_num(node)
            # node.waves = np.zeros(node.num_state)
            # node.waits = np.zeros(node.num_state)

    def _set_phase(self, action, phase_type, phase_duration):
        for node_name, a in zip(self.node_names, list(action)):
            phase = self._get_node_phase(a, node_name, phase_type)
            self.sim.trafficlight.setRedYellowGreenState(node_name, phase)
            self.sim.trafficlight.setPhaseDuration(node_name, phase_duration)

    def _simulate(self, num_step):
        for _ in range(num_step):
            self.sim.simulationStep()
            self.cur_sec += 1
            if self.is_record:
                self._measure_traffic_step()
    
    def _transfer_action(self, action):
        '''Transfer global action to a list of local actions'''
        phase_nums = []
        for node in self.control_node_names:
            phase_nums.append(self.nodes[node].phase_num)
        action_ls = []
        for i in range(len(phase_nums) - 1):
            action, cur_action = divmod(action, phase_nums[i])
            action_ls.append(cur_action)
        action_ls.append(action)
        return action_ls

    def _update_waits(self, action):
        for node_name, a in zip(self.node_names, action):
            red_lanes = set()
            node = self.nodes[node_name]
            for i in self.phase_map.get_red_lanes(node.phase_id, a):
                red_lanes.add(node.lanes_in[i])
            for i in range(len(node.waits)):
                if self.name == 'real_net':
                    # lane = node.ilds_in[i].split(':')[1]
                    lane = node.ilds_in[i]
                else:
                    lane = 'e:' + node.ilds_in[i].split(':')[1]
                if lane in red_lanes:
                    node.waits[i] += self.control_interval_sec
                else:
                    node.waits[i] = 0

    def collect_tripinfo(self):
        # read trip xml, has to be called externally to get complete file
        trip_file = self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))
        tree = ET.ElementTree(file=trip_file)
        for child in tree.getroot():
            cur_trip = child.attrib
            cur_dict = {}
            cur_dict['episode'] = self.cur_episode
            cur_dict['id'] = cur_trip['id']
            cur_dict['depart_sec'] = cur_trip['depart']
            cur_dict['arrival_sec'] = cur_trip['arrival']
            cur_dict['duration_sec'] = cur_trip['duration']
            cur_dict['wait_step'] = cur_trip['waitSteps']
            cur_dict['wait_sec'] = cur_trip['timeLoss']
            self.trip_data.append(cur_dict)
        # delete the current xml
        cmd = 'rm ' + trip_file
        subprocess.check_call(cmd, shell=True)

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.record_stats = record_stats
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []
        if self.record_stats:
            self.state_stat = {}
            for state_name in self.state_names:
                self.state_stat[state_name] = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.agent)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' % (self.name, self.agent)))

    def reset(self, gui=False, test_ind=0):
        # have to terminate previous sim before calling reset
        self._reset_state()
        if self.train_mode:
            seed = self.seed
        else:
            seed = self.test_seeds[test_ind]
        # self._init_sim(gui=True)
        self._init_sim(seed, gui=gui)
        self.cur_sec = 0
        self.cur_episode += 1
        self._init_sim_traffic()
        # next environment random condition should be different
        self.seed += 1
        return self._get_state()

    def terminate(self):
        self.sim.close()

    def step(self, action):

        self._set_phase(action, 'yellow', self.yellow_interval_sec)
        self._simulate(self.yellow_interval_sec)
        rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec
        self._set_phase(action, 'green', rest_interval_sec)
        self._simulate(rest_interval_sec)
        state = self._get_state()
        reward = self._measure_reward_step()
        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True
        
        if self.is_record:
            action_r = ','.join(['%d' % a for a in action])
            cur_control = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'step': self.cur_sec / self.control_interval_sec,
                           'action': action_r,}
            self.control_data.append(cur_control)

        return state, reward, done


    def update_fingerprint(self, policy):
        for node_name, pi in zip(self.node_names, policy):
            self.nodes[node_name].fingerprint = np.array(pi)[:-1]
