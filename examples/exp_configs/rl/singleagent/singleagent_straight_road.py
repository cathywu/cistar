"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
from flow.controllers import RLController, IDMController
from flow.core.params import EnvParams, NetParams, InitialConfig, InFlows, \
                             VehicleParams, SumoParams, SumoLaneChangeParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks import HighwayNetwork
from flow.envs import SingleStraightRoad
from flow.networks.highway import ADDITIONAL_NET_PARAMS
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env


# SET UP PARAMETERS FOR THE SIMULATION

# number of steps per rollout
HORIZON = 2000

# inflow rate on the highway in vehicles per hour
HIGHWAY_INFLOW_RATE = 10800 / 5
# percentage of autonomous vehicles compared to human vehicles on highway
PENETRATION_RATE = 10


# SET UP PARAMETERS FOR THE NETWORK

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    # length of the highway
    "length": 2000,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # number of edges to divide the highway into
    "num_edges": 2
})


# SET UP PARAMETERS FOR THE ENVIRONMENT

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 2.6,
    'max_decel': 4.5,
    'target_velocity': 18.0,
    'local_reward': True,
    'lead_obs': True,
    "terminate_on_wave": False,
    # the environment is not allowed to terminate below this horizon length
    'wave_termination_horizon': 1000,
    # the speed below which we consider a wave to have occured
    'wave_termination_speed': 10.0,
    # whether the vehicle continues to acquire reward after it exits the system. This causes it to have incentive
    # to leave the network in a good state after it leaves
    'reward_after_exit': True
})


# CREATE VEHICLE TYPES AND INFLOWS

vehicles = VehicleParams()
inflows = InFlows()

# human vehicles
vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
    acceleration_controller=(IDMController, {"a": .3, "b": 2.0, "noise": 0.5}),
)

# autonomous vehicles
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}))

# add human vehicles on the highway
inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=int(HIGHWAY_INFLOW_RATE * (1 - PENETRATION_RATE / 100)),
    depart_lane="free",
    depart_speed="23.0",
    name="idm_highway_inflow")

# add autonomous vehicles on the highway
# they will stay on the highway, i.e. they won't exit through the off-ramps
inflows.add(
    veh_type="rl",
    edge="highway_0",
    vehs_per_hour=int(HIGHWAY_INFLOW_RATE * (PENETRATION_RATE / 100)),
    depart_lane="free",
    depart_speed="23.0",
    name="rl_highway_inflow")

# SET UP FLOW PARAMETERS
done_at_exit = True
if additional_env_params['reward_after_exit']:
    done_at_exit = False

flow_params = dict(
    # name of the experiment
    exp_tag='singleagent_highway',

    # name of the flow environment the experiment is running on
    env_name=SingleStraightRoad,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=0,
        sims_per_step=1,  # do not put more than one
        done_at_exit=done_at_exit,
        additional_params=additional_env_params,
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        use_ballistic=True,
        restart_instance=True
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflows,
        additional_params=additional_net_params
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


# SET UP RLLIB MULTI-AGENT FEATURES

create_env, env_name = make_create_env(params=flow_params, version=0)

# register as rllib env
register_env(env_name, create_env)

# multiagent configuration
test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space
