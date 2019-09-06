import numpy as np
from gym.envs.robotics import rotations as r_tool
from numpy.linalg import inv
from ipdb import set_trace
import math
from math import pi,cos,sin,acos


class imaginary_learning:
    def __init__(self,IER_times=1,err_distance = 0.05):
        self.err_distance = err_distance
        self.IER_times = IER_times
    
    def imagine_process_for_episodes_list(self,episodes):
        for _ in range(self.IER_times):
            xs,ys,zs = self.generate_random_point_in_sphere(len(episodes))

            for (obs, acts, goals, achieved_goals) in episodes:
                imagined_goals_list = []
                for goal,offset_x,offset_y,offset_z in zip(goals,xs,ys,zs):
                    imagined_goal = goal + np.ndarray([offset_x,offset_y,offset_z])
                    imagined_goals_list.append(imagined_goal.copy())
                tmp_episodes.append([obs, acts, imagined_goal, achieved_goals])

        for tmp_episode in tmp_episodes:
            episodes.append(tmp_episode)

        return episodes

    def process_goals(self,goals):
        goals_len = goals.shape[0]
        xs,ys,zs = self.generate_random_point_in_sphere(goals_len)

        for i,(offset_x,offset_y,offset_z) in enumerate(zip(xs,ys,zs)):
            goals[i] = goals[i] + np.array([offset_x,offset_y,offset_z])
        return goals.copy()


    # def imagine_process_for_transitions(self,transitions):
    #     for _ in range(self.IER_times):
    #         xs,ys,zs = self.generate_random_point_in_sphere(len(episodes))

    #         for (obs, acts, goals, achieved_goals) in episodes:
    #             imagined_goals_list = []
    #             for goal,offset_x,offset_y,offset_z in zip(goals,xs,ys,zs):
    #                 imagined_goal = goal + np.ndarray([offset_x,offset_y,offset_z])
    #                 imagined_goals_list.append(imagined_goal.copy())
    #             tmp_episodes.append([obs, acts, imagined_goal, achieved_goals])

    #     for tmp_episode in tmp_episodes:
    #         episodes.append(tmp_episode)

    #     return transitions

        
    def generate_random_point_in_sphere(self,goals_len):
        angle1s=np.random.random(size=goals_len)*2*pi
        random_radians = np.random.random(size=goals_len)*2-1

        agnle2s = []
        for rand_rad in random_radians:
            angle2=acos(rand_rad)
            agnle2s.append(angle2)
        agnle2s = np.asarray(agnle2s, dtype=np.float32)
        rs=np.random.random(size=goals_len)**(1/3)

        xs =[]
        ys = []
        zs = []

        for a1,a2,r in zip(angle1s,agnle2s,rs):
            
            x=r*cos(a1)*sin(a2) * self.err_distance
            y=r*sin(a1)*sin(a2) * self.err_distance
            z=r*cos(a2) * self.err_distance
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return xs,ys,zs