"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
import re

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

from flow.core import rewards
from flow.envs.base import Env

ADDITIONAL_ENV_PARAMS = {
    # minimum time that each intersection's traffic lights should remain in a yellow phase for (in seconds)
    "min_yellow_time": 5.0,
    # minimum time that each intersection's traffic lights should remain in a green phase for (in seconds)
    "min_green_time": 20.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
}

ADDITIONAL_PO_ENV_PARAMS = {
    # num of vehicles the agent can observe on each incoming edge TODO(KevinLin) Huh? This would turn out to be useless in the new state observation format? I'll keep it for now.
    "num_observed": 2,
    # velocity to use in reward functions
    "target_velocity": 30,
}
"""
For every edge, legal right turns are given either given priority green light or secondary green light
unless otherwise specified (as per US traffic light rules?)

PHASE DEFINITIONS:

vertical_green = cars on the vertical edges have green lights for going straight are free to go straight
vertical_green_to_yellow = cars on the vertical edges that were free to go straight now face a yellow light
horizontal_green = similar to vertical counterpart
vertical_green_to_yellow = similar to vertical counterpart

protected_left_X = cars on the X edge have a protected left turn i.e. priority greens for the X edge cars turning
left, going straight and turning right. Cars from other edges have red (apart from their secondary green right turns).
protected_left_X_to_yellow = cars on the X edge that were free to go left/straight now face a yellow light

Here, X in [top, right, bottom left]

"""

PHASE_NUM_TO_STR = {0: "vertical_green", 6: "vertical_green_to_yellow",
                    1: "horizontal_green", 7: "horizontal_green_to_yellow",
                    2: "protected_left_top", 8: "protected_left_top_to_yellow",
                    3: "protected_left_right", 9: "protected_left_right_to_yellow",
                    4: "protected_left_bottom", 10: "protected_left_bottom_to_yellow",
                    5: "protected_left_left", 11: "protected_left_left_to_yellow"}
"""
In the case that the RL 
"""
PHASE_REPEAT_PRESET_ORDER = {0: 1,
                          1: 0,
                          2: 3,
                          3: 4,
                          4: 5,
                          5: 2}


class QueueGridEnv(Env):
    """Environment used to train traffic lights.

    Required from env_params:

    * switch_time: minimum time a light must be constant before
      it switches (in seconds).
      Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL,
      options are respectively "actuated" and "controlled"
    * discrete: determines whether the action space is meant to be discrete or
      continuous

    States
        An observation consists of:
        a) the number of cars in each lane
        b) a timer of how long a traffic light has been in its current phase TODO(KevinLin) Either the purely green light phase or both the green and the yellow phase
        c) the current traffic light phase for every intersection that has traffic lights (in this case, that's every intersection)

    Actions
        The action space consists of a list of float variables ranging from 0-1 specifying:
        a) [For a currently 'green' intersection] Whether an intersection should switch to its corresponding yellow phase
        b) [For a currently 'yellow' intersection] The phase that the traffic lights of the intersection should switch to

        Actions are sent to the traffic lights at intersections in the grid from left to right
        and then top to bottom.


        Note: At the end of a 'yellow' phase, the RL agent may output a number that's equivalent to switching back to
        the corresponding green phase intersection (e.g. phase 1 green -> phase 1 yellow -> phase 1 green). Instead of
        allowing this repeat, we manually assign a non-current phase for the new phase. Specifically, we'll use the
        PHASE_REPEAT_PRESET_ORDER dict (given above) to deal with this situation.

    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.

    Attributes
    ----------
    grid_array : dict
        Array containing information on the traffic light grid, such as the
        length of roads, row_num, col_num, number of initial cars
    rows : int
        Number of rows in this traffic light grid network
    cols : int
        Number of columns in this traffic light grid network
    num_tl_intersections : int
        Number of intersections (with traffic lights) in this traffic light grid network
    tl_type : str
        Type of traffic lights, either 'actuated' or 'static'
    steps : int
        Horizon of this experiment, see EnvParams.horizon
    obs_var_labels : dict
        Referenced in the visualizer. Tells the visualizer which
        metrics to track
    node_mapping : dict
        Dictionary mapping intersections / nodes (nomenclature is used
        interchangeably here) to the edges that are leading to said
        intersection / node

    ### For all of the following attributes, each entry in the array corresponds to one particular intersection

    curr_phase_duration : np array [num_tl_intersections]x1 np array
        Multi-dimensional array keeping track, in timesteps, of how much time
        has passed since changing to the current phase
    curr_phase : np array [num_tl_intersections]x1 np array                T
        Multi-dimensional array keeping track of the phase that the traffic lights corresponding to particular
        intersections are currently in. Refer to the "PHASE_NUM_TO_STR" dict above for what a number represents.
    currently_green : np array [num_tl_intersections]x1 np array
        Multi-dimensional array keeping track of whether an intersection of traffic lights is currently in the green
        part of its phase.
        1 if green, 0 if yellow
    min_yellow_time : np array [num_tl_intersections]x1 np array
        The minimum time in timesteps that a light can be yellow. 5s by default.
        Serves as a lower bound.
    min_green_time : np array [num_tl_intersections]x1 np array
        The minimum time in timesteps that a light can be yellow. 20s by default. # This is a somewhat arbitrary choice
        Serves as a lower bound

    ###

    discrete : bool
        Indicates whether or not the action space is discrete. See below for
        more information:
        https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        self.horizontal_lanes = self.network.net_params.additional_params["horizontal_lanes"]
        self.vertical_lanes = self.network.net_params.additional_params["vertical_lanes"]
        self.num_tl_intersections = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'cars_per_lane': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'curr_phase_duration': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'curr_phase': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        # Keeps track of how long the traffic lights in an intersection have been in their current green phase
        # or yellow phase (e.g. when switching from phase 1 green to phase 1 yellow, the timer resets)
        self.curr_phase_duration = np.zeros((self.num_tl_intersections, 1))

        # when this hits min_switch_time, we change from phase x's yellow to phase y's green (where x != y)

        self.min_yellow_time = env_params.additional_params["min_yellow_time"]
        self.min_green_time = env_params.additional_params["min_green_time"]

        # Keeps track of the traffic light phase of each intersection. See phase definitions above.
        self.phases = np.zeros((self.num_tl_intersections, 1))

        # Value of 1 indicates that the intersection is in its phase's green state
        # Value of 0 indicates that the intersection is in its phases' yellow state
        self.currently_green = np.zeros((self.num_tl_intersections, 1))

        x_max = self.cols + 1
        y_max = self.rows + 1

        if self.tl_type != "actuated":
            for x in range(1, x_max):
                for y in range(1, y_max):
                    self.k.traffic_light.set_state(         # TODO: what's this k variable?
                        node_id="({}.{})".format(x, y), state=PHASE_NUM_TO_STR[0])    # TODO(KevinLin): How should the grid be initialized?
                    self.currently_green[y * self.cols + x] = 1

        # # Additional Information for Plotting
        # self.edge_mapping = {"top": [], "bot": [], "right": [], "left": []}
        # for i, veh_id in enumerate(self.k.vehicle.get_ids()):
        #     edge = self.k.vehicle.get_edge(veh_id)
        #     for key in self.edge_mapping:
        #         if key in edge:
        #             self.edge_mapping[key].append(i)
        #             break

        # check whether the action space is meant to be discrete or continuous
        self.discrete = env_params.additional_params.get("discrete", False)

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(2 ** self.num_tl_intersections)
        else:   # TODO(KevinLin) why's this low = -1 and high = 1??
            return Box(
                low=-1,
                high=1,
                shape=(self.num_tl_intersections,),
                dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        cars_per_lane = Box(
            low=0.,
            high=1,
            shape=(self.total_lanes()),      # look at all the lanes
            dtype=np.float32)  # check how discrete values work
        # Check what the low=0, high=1 means
        curr_phase_duration = Box(
            low=0.,
            high=1,
            shape=(self.num_tl_intersections,),
            dtype=np.float32)
        curr_phase = Box(
            low=0.,
            high=1,
            shape=(self.num_tl_intersections,),
            dtype=np.float32)
        currently_green = Box(
            low=0.,
            high=1,
            shape=(self.num_tl_intersections,),
            dtype=np.float32)

        return Tuple((cars_per_lane, curr_phase_duration, curr_phase, currently_green))
        # TODO(KevinLin) any advantage of tuple over one large box?

    def get_state(self):
        """See class definition."""

        # get the state arrays
        cars_per_lane = 20

        state = [
            cars_per_lane,
            self.curr_phase_duration.flatten().tolist(),
            self.curr_phase.flatten().tolist(),
            self.currently_green.flatten().tolist()
        ]

        return np.array(state)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:
            # convert single value to list of 0's and 1's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_tl_intersections - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0 to zero and above 0 to 1. 0 indicates
            # that should not switch the direction, and 1 indicates that switch
            # should happen
            rl_mask = rl_actions > 0.0

        for i, action in enumerate(rl_mask):
            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='rGrG')                                   # TODO: Kevin - found it! Change this thing, and other dependencies
                                                                            # Yikes, looks like I'm going to have fun understanding what Kathy? wrote!
                    self.currently_yellow[i] = 0
            else:
                if action:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='ryry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return - rewards.min_delay_unscaled(self) \
            - rewards.boolean_action_penalty(rl_actions >= 0.5, gain=1.0)

    # ===============================
    # ============ UTILS ============
    # ===============================

    def generate_tl_phases(self, phase_type, horiz_lanes, vert_lanes):
        """Returns the tl phase string for the corresponding phase types.
        Note: right turns will have 'g' by default"""

        if phase_type == "vertical_green":
            vertical = "G" + vert_lanes * "G" + "r"  # right turn, straights, left turn
            horizontal = "g" + horiz_lanes * "r" + "r"  # right turn, straights, left turn
            return vertical + horizontal + vertical + horizontal

        elif phase_type == "vertical_green_to_yellow":
            horizontal = "G" + vert_lanes * "G" + "r"  # right turn, straights, left turn
            vertical = "g" + horiz_lanes * "y" + "r"  # right turn, straights, left turn
            return vertical + horizontal + vertical + horizontal

        elif phase_type == "horizontal_green":
            horizontal = "G" + vert_lanes * "G" + "r"  # right turn, straights, left turn
            vertical = "g" + horiz_lanes * "r" + "r"  # right turn, straights, left turn
            return vertical + horizontal + vertical + horizontal

        elif phase_type == "horizontal_green_to_yellow":
            horizontal = "g" + vert_lanes * "y" + "r"  # right turn, straights, left turn
            vertical = "g" + horiz_lanes * "r" + "r"  # right turn, straights, left turn
            return vertical + horizontal + vertical + horizontal

        elif phase_type == "protected_left_top":
            top = "G" + "G" * vert_lanes + "G"
            bot = "g" + "r" * vert_lanes + "r"
            horizontal = "g" + "r" * horiz_lanes + "r"  # right turn, straights, left turn
            return top + horizontal + bot + horizontal

        elif phase_type == "protected_left_top_to_yellow":
            top = "g" + "y" * vert_lanes + "y"
            bot = "g" + "r" * vert_lanes + "r"
            horizontal = "g" + "r" * horiz_lanes + "r"  # right turn, straights, left turn
            return top + horizontal + bot + horizontal

        elif phase_type == "protected_left_right":
            vertical = "g" + "r" * vert_lanes + "r"
            left = "g" + "r" * horiz_lanes + "r"
            right = "g" + "G" * horiz_lanes + "G"
            return vertical + right + vertical + left

        elif phase_type == "protected_left_right_to_yellow":
            vertical = "g" + "r" * vert_lanes + "r"
            left = "g" + "r" * horiz_lanes + "r"
            right = "g" + "y" * horiz_lanes + "y"
            return vertical + right + vertical + left

        elif phase_type == "protected_left_bottom":
            bot = "G" + "G" * vert_lanes + "G"
            top = "g" + "r" * vert_lanes + "r"
            horizontal = "g" + "r" * horiz_lanes + "r"  # right turn, straights, left turn
            return top + horizontal + bot + horizontal

        elif phase_type == "protected_left_bottom_to_yellow":
            bot = "g" + "y" * vert_lanes + "y"
            top = "g" + "r" * vert_lanes + "r"
            horizontal = "g" + "r" * horiz_lanes + "r"  # right turn, straights, left turn
            return top + horizontal + bot + horizontal

        elif phase_type == "protected_left_left":
            vertical = "g" + "r" * vert_lanes + "r"
            right = "g" + "r" * horiz_lanes + "r"
            left = "g" + "G" * horiz_lanes + "G"
            return vertical + right + vertical + left

        elif phase_type == "protected_left_left_to_yellow":
            vertical = "g" + "r" * vert_lanes + "r"
            right = "g" + "r" * horiz_lanes + "r"
            left = "g" + "y" * horiz_lanes + "y"
            return vertical + right + vertical + left

    def total_lanes(self):
        """Computes the total number of lanes in a queue grid"""
        horiz_lanes = self.horizontal_lanes
        vert_lanes = self.vertical_lanes
        print("TO IMPLEMENT - KEVINLIN")
        return horiz_lanes * vert_lanes


    def _convert_edge(self, edges):
        """Convert the string edge to a number.

        Start at the bottom left vertical edge and going right and then up, so
        the bottom left vertical edge is zero, the right edge beside it  is 1.

        The numbers are assigned along the lowest column, then the lowest row,
        then the second lowest column, etc. Left goes before right, top goes
        before bottom.

        The values are zero indexed.

        Parameters
        ----------
        edges : list of str or str
            name of the edge(s)

        Returns
        -------
        list of int or int
            a number uniquely identifying each edge
        """
        if isinstance(edges, list):
            return [self._split_edge(edge) for edge in edges]
        else:
            return self._split_edge(edges)

    def _split_edge(self, edge):
        """Act as utility function for convert_edge."""
        if edge:
            if edge[0] == ":":  # center
                center_index = int(edge.split("center")[1][0])          # TODO: Kevin Yikes, change this too
                base = ((self.cols + 1) * self.rows * 2) \
                    + ((self.rows + 1) * self.cols * 2)
                return base + center_index + 1
            else:
                pattern = re.compile(r"[a-zA-Z]+")
                edge_type = pattern.match(edge).group()
                edge = edge.split(edge_type)[1].split('_')
                row_index, col_index = [int(x) for x in edge]
                if edge_type in ['bot', 'top']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * (row_index + 1))
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'bot' else edge_num + 1
                if edge_type in ['left', 'right']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * row_index)
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'left' else edge_num + 1
        else:
            return 0

    def _get_relative_node(self, agent_id, direction):
        """Yield node number of traffic light agent in a given direction.

        For example, the nodes in a traffic light grid with 2 rows and 3
        columns are indexed as follows:

            |     |     |
        --- 3 --- 4 --- 5 ---
            |     |     |
        --- 0 --- 1 --- 2 ---
            |     |     |

             |       |     |
        --- 2.1 --- 2.2 --- 3.2 ---
             |       |       |
        --- 1.1 --- 1.2 --- 1.3 ---
             |       |       |

        TODO(kevin) remove this^

        See flow.networks.traffic_light_grid for more information.

        Example of function usage:
        - Seeking the "top" direction to ":(1.1)" would return 3.
        - Seeking the "bottom" direction to :(1.1)" would return -1.

        Parameters
        ----------
        agent_id : str
            agent id of the form ":({}.{})".format(x_pos, y_pos)
        direction : str
            top, bottom, left, right

        Returns
        -------
        int
            node number # Nodes without traffic lights yield -1
        """
        # TODO(Kevin Lin) what's the point of the colon here?

        agent_node_coords = [agent_id[i] for i in range(len(agent_id)) if agent_id[i].isdigit()]
        agent_node_x, agent_node_y = int(agent_node_coords[0]), int(agent_node_coords[1])
        agent_id_num = (agent_node_x - 1) + (agent_node_y - 1) * self.cols

        if direction == "top":
            node = agent_id_num + self.cols
            if node >= self.num_tl_intersections:
                node = -1
        elif direction == "bottom":
            node = agent_id_num - self.cols
            if node < 0:
                node = -1
        elif direction == "left":
            if agent_id_num % self.cols == 0:
                node = -1
            else:
                node = agent_id_num - 1
        elif direction == "right":
            if agent_id_num % self.cols == self.cols - 1:
                node = -1
            else:
                node = agent_id_num + 1
        else:
            raise NotImplementedError

        return node

    def additional_command(self):
        """See parent class.

        Used to insert vehicles that are on the exit edge and place them
        back on their entrance edge.
        """
        for veh_id in self.k.vehicle.get_ids():
            self._reroute_if_final_edge(veh_id)

    def _reroute_if_final_edge(self, veh_id):
        """Reroute vehicle associated with veh_id.

        Checks if an edge is the final edge. If it is return the route it
        should start off at.
        """
        edge = self.k.vehicle.get_edge(veh_id)
        if edge == "":
            return
        if edge[0] == ":":  # center edge
            return
        pattern = re.compile(r"[a-zA-Z]+")
        edge_type = pattern.match(edge).group()
        edge = edge.split(edge_type)[1].split('_')
        row_index, col_index = [int(x) for x in edge]

        # find the route that we're going to place the vehicle on if we are
        # going to remove it
        route_id = None
        if edge_type == 'bot' and col_index == self.cols:       # TODO: Kevin :) Change these to new scheme
            route_id = "bot{}_0".format(row_index)
        elif edge_type == 'top' and col_index == 0:
            route_id = "top{}_{}".format(row_index, self.cols)
        elif edge_type == 'left' and row_index == 0:
            route_id = "left{}_{}".format(self.rows, col_index)
        elif edge_type == 'right' and row_index == self.rows:
            route_id = "right0_{}".format(col_index)

        if route_id is not None:
            type_id = self.k.vehicle.get_type(veh_id)
            lane_index = self.k.vehicle.get_lane(veh_id)
            # remove the vehicle
            self.k.vehicle.remove(veh_id)
            # reintroduce it at the start of the network
            self.k.vehicle.add(
                veh_id=veh_id,
                edge=route_id,
                type_id=str(type_id),
                lane=str(lane_index),
                pos="0",
                speed="max")


class QueueGridPOEnv(QueueGridEnv):
    """Environment used to train traffic lights.

    Required from env_params:

    * switch_time: minimum switch time for each traffic light (in seconds).
      Earlier RL commands are ignored.


    States
        An observation is the number of observed vehicles in each intersection
        closest to the traffic lights, a number uniquely identifying which
        edge the vehicle is on, and the speed of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the delay of each vehicle minus a penalty for switching
        traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.

    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_PO_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 2)

        # used during visualization
        self.observed_ids = []

    @property
    def observation_space(self):
        """State space that is partially observed.

        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, edge information, and traffic light
        state.
        """
        tl_box = Box(
            low=0.,
            high=1,
            shape=(3 * 4 * self.num_observed * self.num_tl_intersections +
                   2 * len(self.k.network.get_edge_list()) +
                   3 * self.num_tl_intersections,),
            dtype=np.float32)
        return tl_box

    def get_state(self):            # TODO: Density?
        """See parent class.

        Returns self.num_observed number of vehicles closest to each traffic
        light and for each vehicle its velocity, distance to intersection,
        edge_number traffic light state. This is partially observed


        """

        speeds = []
        dist_to_intersec = []
        edge_number = []
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        all_observed_ids = []

        for _, edges in self.network.node_mapping:
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection(edge, self.num_observed)
                all_observed_ids += observed_ids

                # check which edges we have so we can always pad in the right
                # positions
                speeds += [
                    self.k.vehicle.get_speed(veh_id) / max_speed
                    for veh_id in observed_ids
                ]
                dist_to_intersec += [
                    (self.k.network.edge_length(
                        self.k.vehicle.get_edge(veh_id)) -
                        self.k.vehicle.get_position(veh_id)) / max_dist
                    for veh_id in observed_ids
                ]
                edge_number += \
                    [self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
                     (self.k.network.network.num_edges - 1)
                     for veh_id in observed_ids]

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    speeds += [0] * diff
                    dist_to_intersec += [0] * diff
                    edge_number += [0] * diff

        # now add in the density and average velocity on the edges
        density = []
        velocity_avg = []
        for edge in self.k.network.get_edge_list():
            ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(ids) > 0:
                vehicle_length = 5
                density += [vehicle_length * len(ids) /
                            self.k.network.edge_length(edge)]
                velocity_avg += [np.mean(
                    [self.k.vehicle.get_speed(veh_id) for veh_id in
                     ids]) / max_speed]
            else:
                density += [0]
                velocity_avg += [0]
        self.observed_ids = all_observed_ids
        return np.array(
            np.concatenate([
                speeds, dist_to_intersec, edge_number, density, velocity_avg,
                self.last_change.flatten().tolist(),
                self.direction.flatten().tolist(),
                self.currently_yellow.flatten().tolist()
            ]))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return (- rewards.min_delay_unscaled(self) +
                    rewards.penalize_standstill(self, gain=0.2))

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        [self.k.vehicle.set_observed(veh_id) for veh_id in self.observed_ids]


class QueueGridTestEnv(QueueGridEnv):
    """
    Class for use in testing.

    This class overrides RL methods of traffic light grid so we can test
    construction without needing to specify RL methods
    """

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """No return, for testing purposes."""
        return 0
