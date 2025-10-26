'''
    This RCCarWrapper is for 
        1. usage issue (gymnasium style)
        2. define observation & action space
        3. observation processing (concatenate lidar + other properties)
        
        ->  outputs of step function should have (n_agents, dim) shape
            ex) observations, rewards, terminates, truncates, infos = env.step([agent_1_action, 
                                                                                agent_2_action,
                                                                                ,,,
                                                                                agent_n_action])
'''

import numpy as np
import gymnasium as gym
import rccar_gym

class RCCarWrapper(gym.Wrapper):
    def __init__(self, args=None, maps=None, render_mode=None) -> None:
        self._env = gym.make("rccar-v0", 
                            args=args,
                            maps=maps,
                            render_mode=render_mode)
        super().__init__(self._env)
        
        start, end = (np.array(args.lidar_range) * 4 + 540).astype(int)      # degree to scan index
        self.start, self.end = max(0, start), min(1080, end)

        #####
        self.track = self._env.unwrapped.track
        self.waypoints = np.stack([self.track.centerline.xs, self.track.centerline.ys]).T 
        self.N = self.waypoints.shape[0]
        self.checkpoints = [] 
        self.n_checkpoints = 20
        self.next_checkpoint = 0 
        self.arrive_distance = 1.5
        #####
        
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer

    def reset(self, **kwargs):
        obs_dict, info = self._env.reset(**kwargs)

        pos_x, pos_y, yaw = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        self.pos = np.stack([pos_x, pos_y])
        self.yaw = yaw
        self.scan = obs_dict['scans'][0]

        self.scan = self.scan[self.start:self.end]
        
        info['obs_dict'] = obs_dict

        #####
        dpos = self.waypoints - self.pos
        dists = np.linalg.norm(dpos, axis=-1)
        init_idx = np.argmin(dists)
        for i in range(self.n_checkpoints):
            self.checkpoints.append(self.waypoints[(int(self.N*(i+1)/self.n_checkpoints)-1+init_idx)%self.N])
        #####

        return [self.pos, self.yaw, self.scan], info

    def step(self, action:np.array):
        
        steer = action[0]
        speed = action[1]

        steer = np.clip(steer, -self.max_steer, self.max_steer)
        speed = np.clip(speed, self.min_speed, self.max_speed)
        
        clipped_action = np.array([steer, speed])
        
        obs_dict, _, terminate, truncate, info = self._env.step(clipped_action)

        pos_x, pos_y, yaw = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0] # rad
        self.pos = np.stack([pos_x, pos_y])
        self.yaw = yaw
        self.scan = obs_dict['scans'][0]
        
        self.scan = self.scan[self.start:self.end]

        info['obs_dict'] = obs_dict
        reward = 0.0

        #####
        if self.next_checkpoint != self.n_checkpoints:
            checkpoint_pos = self.checkpoints[self.next_checkpoint]
            if np.linalg.norm(self.pos - checkpoint_pos) < self.arrive_distance:
                self.next_checkpoint += 1
        info['waypoint'] = self.next_checkpoint
        #####
        return [self.pos, self.yaw, self.scan], reward, terminate, truncate, info
    
    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
