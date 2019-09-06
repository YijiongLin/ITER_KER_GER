import numpy as np
from gym.envs.robotics import rotations as r_tool
from numpy.linalg import inv
from ipdb import set_trace
import math

PI = math.pi
SYM_PLANE_Y = 0.75 * 2

IF_CLEAR_BUFFER = False
SINGLE_SUC_RATE_THRESHOLD = None  # Set to none if donnot terminate KER


MAX_Z_THETA_PICK_PUSH = 0.1443
MAX_Z_THETA_SLIDE = 0.0697
COUNT_UNVALID_OBJ = True
unvalid_episode = False
BOOL_OUTPUT_ONE_EPISODE_TRAJ = False # Generated one episode KER trajectories for plotting
IER_TIMES = 3
ERR_DISTANCE = 0.05
class mirror_learning:
    def __init__(self,env_type,n_rsym):
        self.env_type = env_type
        self.n_rsym = n_rsym
        self.sym_plane = None
        if (self.env_type == 'FetchPickAndPlace-v1') or (self.env_type == 'FetchPush-v1' )or (self.env_type == 'FetchReach-v1' ) :
            self.max_z_theta= MAX_Z_THETA_PICK_PUSH
            self.robot_base_x = 0.695
            self.robot_base_y = 0.75
        elif  self.env_type == 'FetchSlide-v1' :
            self.max_z_theta = MAX_Z_THETA_SLIDE
            self.robot_base_x = 0.34
            self.robot_base_y = 0.75

    def y_mirror(self,param):
        return self.sym_plane_compute(param,'y_axis','y_mirror')

    def x_mirror():
        self.sym_plane_compute(param,'x_axis','y_mirror')


    def kaleidoscope_robot(self, param, z_theta, sym_axis = 'y_axis', sym_method = 'y_mirror'):
        
        if sym_axis == 'y_axis':
            # in linear variable, plus i; in angular variable, minus i
            i = 0
        elif sym_axis == 'x_axis':
            i = -1

        if sym_method == 'y_mirror':
            self.sym_plane = SYM_PLANE_Y
        elif sym_method == 'x_mirror':
            self.sym_plane = SYM_PLANE_X

        # compute the rotation transformation & its inverse.
        theta = np.array([0,0,z_theta])
        rot_z_theta = r_tool.euler2mat(theta)

        inv_theta = np.array([0,0,-z_theta])
        inv_rot_z_theta = r_tool.euler2mat(inv_theta)
        # Determine the input is which element
        param_len = len(param[0])
        #goal & achieved goal  
        if param_len == 3:    
            v_l_a = param[0][0:3]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(True, v_l_a, rot_z_theta, inv_rot_z_theta, i)
            param[0][0:3] =  s_v_l_a

        elif param_len == 4:  #action
            v_l_a = param[0][0:3]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(False, v_l_a, rot_z_theta, inv_rot_z_theta, i)
            param[0][0:3] =  s_v_l_a

        elif param_len == 10:     # observation without object
            # grip pos
            v_l_a = param[0][0:3]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(True, v_l_a, rot_z_theta, inv_rot_z_theta, i)
            param[0][0:3] =  s_v_l_a
            # grip vel
            v_l_a = param[0][3:6]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(False, v_l_a, rot_z_theta, inv_rot_z_theta, i)
            param[0][3:6] =  s_v_l_a
            
        elif param_len == 25:     # observation with object
            # sym_grip_pos
            v_l_a = param[0][0:3]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(True, v_l_a, rot_z_theta, inv_rot_z_theta, i)
            param[0][0:3] =  s_v_l_a

            # sym_obj_pos
            v_l_a = param[0][3:6]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(True, v_l_a, rot_z_theta, inv_rot_z_theta, i)
            param[0][3:6] =  s_v_l_a

            # sym_obj_rel_pos
            param[0][6] = param[0][3]-param[0][0]
            param[0][7] = param[0][4]-param[0][1]
            param[0][8] = param[0][5]-param[0][2]

            # sym_obj_rot_euler
            theta_a = param[0][11:14]
            param[0][11:14] = self.orientation_mat_symmetric_with_rot_plane(theta_a, rot_z_theta, inv_rot_z_theta, i)
            
            # get the obj real velp 
            v_l_a = param[0][14:17]+param[0][20:23]
            # get the sym obj real velp
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(False, v_l_a, rot_z_theta, inv_rot_z_theta, i)
            param[0][14:17] =  s_v_l_a
            # get the sym grip real velp
            v_l_a = param[0][20:23]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(False, v_l_a, rot_z_theta, inv_rot_z_theta, i)
            param[0][20:23] =  s_v_l_a
            # get the sym obj relative velp 
            param[0][14:17] -= param[0][20:23]

            # sym_obj_velr
            theta_a = param[0][17:20]
            param[0][17:20] = self.orientation_mat_symmetric_with_rot_plane(theta_a, rot_z_theta, inv_rot_z_theta, i)
        
        return param.copy()


    def linear_vector_symmetric_with_rot_plane(self,if_pos, v_l_a, rot_z_theta, inv_rot_z_theta, i):
        # Point 'a' position = v_l_a
        if if_pos == True:
            v_l_a[0] -= self.robot_base_x
            v_l_a[1] -= self.robot_base_y
        v_l_a_hat = np.dot(inv_rot_z_theta,v_l_a)
        v_l_a_hat[1+i] = -v_l_a_hat[1+i]
        s_v_l_a =  np.dot(rot_z_theta,v_l_a_hat)
        if if_pos == True:
            s_v_l_a[0] += self.robot_base_x
            s_v_l_a[1] += self.robot_base_y
        return s_v_l_a.copy()


    def orientation_mat_symmetric_with_rot_plane(self, theta_a, rot_z_theta, inv_rot_z_theta, i):
        # Point 'a' orientation euler angle = theta_a
        # euler to rot matrix for transform
        v_r_a = r_tool.euler2mat(theta_a)
        # transform to the O cordinate from S cordinate
        v_r_a_hat = np.matmul(v_r_a,inv_rot_z_theta)

        # Rot matrix to euler for sym
        v_r_a_hat = r_tool.mat2euler(v_r_a_hat)

        # Sym on transformed S's xoz plane (y axis)
        v_r_a_hat[0-i] = -v_r_a_hat[0-i]
        v_r_a_hat[2] = -v_r_a_hat[2]

        # euler to rot matrix for transform 
        v_r_a_hat = r_tool.euler2mat(v_r_a_hat)
        s_v_r_a = np.matmul(v_r_a_hat,rot_z_theta)

        # Rot matrix to euler for return
        s_v_r_a = r_tool.mat2euler(s_v_r_a)
        return s_v_r_a.copy()

    def kaleidoscope_obj():
        pass
    def mirage():
        pass


    def sym_plane_compute(self,param,sym_axis,sym_method):
        # This function is for the vanilla mirror. (x&y mirror)
        if sym_axis == 'y_axis':
            # in linear variable, plus i; in angular variable, minus i
            i = 0
        elif sym_axis == 'x_axis':
            i = -1


        if sym_method == 'y_mirror':
            self.sym_plane = SYM_PLANE_Y
        elif sym_method == 'x_mirror':
            self.sym_plane = SYM_PLANE_X
        # elif sym_method == 'kaleidoscope_robot':
        #     SYM_PLANE = SYM_PLANE_Y


        param_len = len(param[0])
        if param_len == 3:    #goal & achieved goal
            param[0][1+i] = self.sym_plane - param[0][1+i]

        elif param_len == 4:  #action
            param[0][1+i] = - param[0][1+i]

        elif param_len == 10:     # observation without object
            param[0][1+i] = self.sym_plane - param[0][1+i]
            # vel do not need SYM_PLANE
            param[0][4+i] = -param[0][4+i]
            
        elif param_len == 25:     # observation with object

            # sym_grip_pos
            param[0][1+i] = self.sym_plane - param[0][1+i]
            # sym_obj_pos
            param[0][4+i] = self.sym_plane - param[0][4+i]

            # sym_obj_rel_pos
            param[0][6] = param[0][3]-param[0][0]
            param[0][7] = param[0][4]-param[0][1]
            param[0][8] = param[0][5]-param[0][2]

            # sym_obj_rot_euler
            param[0][11-i] = -param[0][11-i]
            param[0][13] = -param[0][13]

            # get the sym_obj_rel_velp & sym_grip_velp
            # no need to transform back to original pose, it can be directly compute the relative pose.
            param[0][15+i] = -param[0][15+i]
            param[0][21+i] = -param[0][21+i]
            
            # sym_obj_velr
            param[0][17-i] = -param[0][17-i]
            param[0][19] = -param[0][19]
        return param.copy()


    def mirror_process(self,obs,acts,goals,achieved_goals):


        # ---------------------------recursive symmetry------------------------------------------------
        ka_episodes_set=[]
        ka_episodes_set.append([obs,acts,goals,achieved_goals])
        z_theta_set = []

        # If self.n_rsym == None, means use vanillar her, or in test mode.
        if self.n_rsym == None or self.n_rsym == 0:
            if BOOL_OUTPUT_ONE_EPISODE_TRAJ:
                np.save(('/home/bourne/data_plot/all_n_rsym_trajs/sym_'+str(self.n_rsym)+'.npy'), ka_episodes_set)
                set_trace()
            return ka_episodes_set


        # not finished yet
        # self.compute_sym_number(goals[0][0])

        # One symmetry will be done in the y mirror, so here n_rsym need to minus 1 
        for _ in range(self.n_rsym-1):
            z_theta = np.random.uniform(0, self.max_z_theta)
            z_theta_set.append(z_theta)

        if BOOL_OUTPUT_ONE_EPISODE_TRAJ:
            #output the symmetric thetas for one step 
            output_theta_set = z_theta_set.copy()
            output_theta_set.append(0)
            save_dir = '/home/bourne/data_plot/all_n_rsym_thetas/thetas_n_rsym_'+str(self.n_rsym)+'.npy'
            np.save(save_dir, output_theta_set)

        for z_theta in z_theta_set:
            ka_episodes_tem = []
            for [o_obs, o_acts, o_goals, o_achieved_goals] in ka_episodes_set:
                s_goals = []
                s_obs = []
                s_acts = []
                s_achieved_goals = []
                for goal in o_goals:
                    s_goal = self.kaleidoscope_robot(goal.copy(),z_theta)
                    s_goals.append(s_goal.copy())

                for ob in o_obs:
                    s_ob = self.kaleidoscope_robot(ob.copy(),z_theta)
                    s_obs.append(s_ob.copy())

                for act in o_acts:
                    s_act = self.kaleidoscope_robot(act.copy(),z_theta)
                    s_acts.append(s_act.copy())

                for achieved_goal in o_achieved_goals:
                    s_achieved_goal = self.kaleidoscope_robot(achieved_goal.copy(),z_theta)
                    s_achieved_goals.append(s_achieved_goal.copy())

                ka_episodes_tem.append([s_obs, s_acts, s_goals, s_achieved_goals])
            for ka_episode in ka_episodes_tem:
                ka_episodes_set.append(ka_episode)
        # ---------------------------end

        #--------------- All datas are symmetrized with y axis.
        ymirror_episode_set = []
        for [o_obs, o_acts, o_goals, o_achieved_goals] in ka_episodes_set:
            y_goals = []
            y_obs = []
            y_acts = []
            y_achieved_goals = []
            for goal in o_goals:
                y_goal = self.y_mirror(goal.copy())
                y_goals.append(y_goal.copy())

            for ob in o_obs:
                y_ob = self.y_mirror(ob.copy())
                y_obs.append(y_ob.copy())

            for act in o_acts:
                y_act = self.y_mirror(act.copy())
                y_acts.append(y_act.copy())

            for achieved_goal in o_achieved_goals:
                y_achieved_goal = self.y_mirror(achieved_goal.copy())
                y_achieved_goals.append(y_achieved_goal.copy())
            ymirror_episode_set.append([y_obs, y_acts, y_goals, y_achieved_goals])
        for ymirror_episode in ymirror_episode_set:
            ka_episodes_set.append(ymirror_episode)

        # output the trajs for one step
        if BOOL_OUTPUT_ONE_EPISODE_TRAJ:
            np.save(('/home/bourne/data_plot/all_n_rsym_trajs/trajs_n_rsym_'+str(self.n_rsym)+'.npy'), ka_episodes_set)
            set_trace()

        return ka_episodes_set
        #--------------- end.

    # def compute_sym_number(self,goal):
    #     self.n_rsym = N_RSYM
        







