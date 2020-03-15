"""Environment for training vehicles to reduce congestion in the I210."""

from gym.spaces import Box, Dict
import numpy as np

from flow.controllers.velocity_controllers import FollowerStopper
from flow.core.params import SumoCarFollowingParams
from flow.core.rewards import average_velocity
from flow.envs.multiagent.base import MultiEnv

# largest number of lanes on any given edge in the network
MAX_LANES = 6

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
    # whether we use an obs space that contains adjacent lane info or just the lead obs
    "lead_obs": True,
    # if imitating, this is how many rollouts to use the expert on without using the agent
    "num_imitation_iters": 1,
    # desired velocity of the follower stopper
    "v_des": 15.0,
    # whether to add in a reward for the speed of nearby vehicles
    "local_reward": True
}


class I210MultiEnv(MultiEnv):
    """Partially observable multi-agent environment for the I-210 subnetworks.

    The policy is shared among the agents, so there can be a non-constant
    number of RL vehicles throughout the simulation.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2

    The following states, actions and rewards are considered for one autonomous
    vehicle only, as they will be computed in the same way for each of them.

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicles in
        all of the preceding lanes as well, a binary value indicating which of
        these vehicles is autonomous, and the speed of the autonomous vehicle.
        Missing vehicles are padded with zeros.

    Actions
        The action consists of an acceleration, bound according to the
        environment parameters, as well as three values that will be converted
        into probabilities via softmax to decide of a lane change (left, none
        or right). NOTE: lane changing is currently not enabled. It's a TODO.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity specified in the environment parameters, while
        slightly penalizing small time headways among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.lead_obs = env_params.additional_params.get("lead_obs")

    @property
    def observation_space(self):
        """See class definition."""
        # speed, speed of leader, headway
        if self.lead_obs:
            return Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(3,),
                dtype=np.float32
            )
        # speed, dist to ego vehicle, binary value which is 1 if the vehicle is
        # an AV
        else:
            leading_obs = 3 * MAX_LANES
            follow_obs = 3 * MAX_LANES

            # speed and lane
            self_obs = 2

            return Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(leading_obs + follow_obs + self_obs,),
                dtype=np.float32
            )

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),  # (4,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                accel = actions[0]

                # lane_change_softmax = np.exp(actions[1:4])
                # lane_change_softmax /= np.sum(lane_change_softmax)
                # lane_change_action = np.random.choice([-1, 0, 1],
                #                                       p=lane_change_softmax)

                self.k.vehicle.apply_acceleration(rl_id, accel)
                # self.k.vehicle.apply_lane_change(rl_id, lane_change_action)

    def get_state(self):
        """See class definition."""
        if self.lead_obs:
            veh_info = {}
            for rl_id in self.k.vehicle.get_rl_ids():
                speed = self.k.vehicle.get_speed(rl_id)
                headway = self.k.vehicle.get_headway(rl_id)
                lead_speed = self.k.vehicle.get_speed(self.k.vehicle.get_leader(rl_id))
                if lead_speed == -1001:
                    lead_speed = 0
                veh_info.update({rl_id: np.array([speed / 50.0, headway / 1000.0, lead_speed / 50.0])})
        else:
            veh_info = {rl_id: np.concatenate((self.state_util(rl_id),
                                               self.veh_statistics(rl_id)))
                        for rl_id in self.k.vehicle.get_rl_ids()}
        return veh_info

    def compute_reward(self, rl_actions, **kwargs):
        # TODO(@evinitsky) we need something way better than this. Something that adds
        # in notions of local reward
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        if self.env_params.additional_params["local_reward"]:
            for rl_id in self.k.vehicle.get_rl_ids():
                speeds = []
                lead_speed = self.k.vehicle.get_speed(self.k.vehicle.get_lane_leaders(rl_id))
                speeds.extend([speed for speed in lead_speed if speed != -1001])
                follow_speed = self.k.vehicle.get_speed(self.k.vehicle.get_lane_followers(rl_id))
                speeds.extend([speed for speed in follow_speed if speed != -1001])
                speeds.append(self.k.vehicle.get_speed(rl_id))
                rewards[rl_id] = np.mean(speeds)

        else:
            for rl_id in self.k.vehicle.get_rl_ids():
                if self.env_params.evaluate:
                    # reward is speed of vehicle if we are in evaluation mode
                    reward = self.k.vehicle.get_speed(rl_id)
                elif kwargs['fail']:
                    # reward is 0 if a collision occurred
                    reward = 0
                else:
                    # reward high system-level velocities
                    cost1 = average_velocity(self, fail=kwargs['fail'])

                    # penalize small time headways
                    cost2 = 0
                    t_min = 1  # smallest acceptable time headway

                    lead_id = self.k.vehicle.get_leader(rl_id)
                    if lead_id not in ["", None] \
                            and self.k.vehicle.get_speed(rl_id) > 0:
                        t_headway = max(
                            self.k.vehicle.get_headway(rl_id) /
                            self.k.vehicle.get_speed(rl_id), 0)
                        cost2 += min((t_headway - t_min) / t_min, 0)

                    # weights for cost1, cost2, and cost3, respectively
                    eta1, eta2 = 1.00, 0.10

                    reward = max(eta1 * cost1 + eta2 * cost2, 0)

                rewards[rl_id] = reward
        return rewards

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)
            # follower
            follow_id = self.k.vehicle.get_follower(rl_id)
            if follow_id:
                self.k.vehicle.set_observed(follow_id)

    def state_util(self, rl_id):
        """Return an array of headway, tailway, leader speed, follower speed.

        Also return a 1 if leader is rl 0 otherwise, a 1 if follower is rl 0 otherwise.
        If there are fewer than MAX_LANES the extra
        entries are filled with -1 to disambiguate from zeros.
        """
        veh = self.k.vehicle
        lane_headways = veh.get_lane_headways(rl_id).copy()
        lane_tailways = veh.get_lane_tailways(rl_id).copy()
        lane_leader_speed = veh.get_lane_leaders_speed(rl_id).copy()
        lane_follower_speed = veh.get_lane_followers_speed(rl_id).copy()
        leader_ids = veh.get_lane_leaders(rl_id).copy()
        follower_ids = veh.get_lane_followers(rl_id).copy()
        rl_ids = self.k.vehicle.get_rl_ids()
        is_leader_rl = [1 if l_id in rl_ids else 0 for l_id in leader_ids]
        is_follow_rl = [1 if f_id in rl_ids else 0 for f_id in follower_ids]
        diff = MAX_LANES - len(is_leader_rl)
        if diff > 0:
            # the minus 1 disambiguates missing cars from missing lanes
            lane_headways += diff * [-1]
            lane_tailways += diff * [-1]
            lane_leader_speed += diff * [-1]
            lane_follower_speed += diff * [-1]
            is_leader_rl += diff * [-1]
            is_follow_rl += diff * [-1]
        lane_headways = np.asarray(lane_headways) / 1000
        lane_tailways = np.asarray(lane_tailways) / 1000
        lane_leader_speed = np.asarray(lane_leader_speed) / 100
        lane_follower_speed = np.asarray(lane_follower_speed) / 100
        return np.concatenate((lane_headways, lane_tailways, lane_leader_speed,
                               lane_follower_speed, is_leader_rl,
                               is_follow_rl))

    def veh_statistics(self, rl_id):
        """Return speed, edge information, and x, y about the vehicle itself."""
        speed = self.k.vehicle.get_speed(rl_id) / 100.0
        lane = (self.k.vehicle.get_lane(rl_id) + 1) / 10.0
        return np.array([speed, lane])


class I210MultiImitationEnv(I210MultiEnv):
    """Imitate a follower stopper controller."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.iter_num = 0
        self.num_imitation_iters = env_params.additional_params.get("num_imitation_iters")
        self.curr_rl_vehicles = {}

    def init_decentral_controller(self, rl_id):
        """Construct a FollowerStopper for each car that we can use as an expert."""
        return FollowerStopper(rl_id, car_following_params=SumoCarFollowingParams(),
                               v_des=self.env_params.additional_params.get("v_des"))

    def update_curr_rl_vehicles(self):
        """Add an additional expert for newly arrived RL vehicles."""
        self.curr_rl_vehicles.update({rl_id: {'controller': self.init_decentral_controller(rl_id)}
                                      for rl_id in self.k.vehicle.get_rl_ids()
                                      if rl_id not in self.curr_rl_vehicles.keys()})

    def set_iteration_num(self, iter_num):
        """Track which training iteration we are on so we know when to turn the expert off."""
        self.iter_num = iter_num

    @property
    def observation_space(self):
        """See parent class."""
        obs = super().observation_space
        return Dict({"a_obs": obs, "expert_action": self.action_space})

    def reset(self, new_inflow_rate=None):
        """See parent class."""
        self.curr_rl_vehicles = {}
        self.update_curr_rl_vehicles()

        state_dict = super().reset(new_inflow_rate)
        return state_dict

    def get_state(self, rl_actions=None):
        """See parent class."""
        # iterate through the RL vehicles and find what the other agent would have done
        self.update_curr_rl_vehicles()

        state_dict = super().get_state()

        return_dict = {}
        for key, value in state_dict.items():
            # this could be the fake final state for vehicles that have left the system
            if key in self.k.vehicle.get_ids():
                controller = self.curr_rl_vehicles[key]['controller']
                accel = controller.get_accel(self)
                # we are on an internal edge and don't take actions
                if accel is None:
                    continue

                return_dict[key] = {"a_obs": value,
                                    "expert_action": np.array([np.clip(accel, a_min=self.action_space.low[0],
                                                                       a_max=self.action_space.high[0])])}
            else:
                # this is just for resetting
                return_dict[key] = {"a_obs": value[:-1], "expert_action": np.array([0.0])}
        # import ipdb; ipdb.set_trace()
        return return_dict

    def _apply_rl_actions(self, rl_actions):
        """See parent class."""
        # iterate through the RL vehicles and find what the other agent would have done
        self.update_curr_rl_vehicles()
        if rl_actions:
            if self.iter_num < self.num_imitation_iters:
                id_list = []
                action_list = []
                for key, value in rl_actions.items():

                    # a vehicle may have left since we got the state
                    if key not in self.k.vehicle.get_arrived_ids() and key in self.k.vehicle.get_rl_ids():
                        controller = self.curr_rl_vehicles[key]['controller']
                        accel = controller.get_accel(self)
                        id_list.append(key)
                        if not accel:
                            accel = -np.abs(self.action_space.low[0])
                        action_list.append(accel)
                self.k.vehicle.apply_acceleration(id_list, action_list)
            else:
                super()._apply_rl_actions(rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See parent class."""
        reward_dict = super().compute_reward(rl_actions)
        new_reward_dict = {}
        for key, value in reward_dict.items():
            controller = self.curr_rl_vehicles[reward_dict]['controller']
            accel = controller.get_accel(self)
            # we are on an internal edge and don't take actions
            if accel is None:
                continue
            else:
                new_reward_dict[key] = value
