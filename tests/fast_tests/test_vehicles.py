import unittest
import os
import numpy as np

from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams, NetParams, \
    InitialConfig, SumoParams, SumoLaneChangeParams
from flow.controllers.car_following_models import IDMController, \
    SimCarFollowingController
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.rlcontroller import RLController

from tests.setup_scripts import ring_road_exp_setup, highway_exp_setup

os.environ["TEST_FLAG"] = "True"


class TestVehiclesClass(unittest.TestCase):
    """
    Tests various functions in the vehicles class
    """

    def test_speed_lane_change_modes(self):
        """
        Check to make sure vehicle class correctly specifies lane change and
        speed modes
        """
        vehicles = VehicleParams()
        vehicles.add(
            "typeA",
            acceleration_controller=(IDMController, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode='obey_safe_speed',
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode="no_lc_safe",
            )
        )

        self.assertEqual(vehicles.type_parameters["typeA"][
                             "car_following_params"].speed_mode, 1)
        self.assertEqual(vehicles.type_parameters["typeA"][
                             "lane_change_params"].lane_change_mode, 512)

        vehicles.add(
            "typeB",
            acceleration_controller=(IDMController, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode='aggressive',
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode="strategic",
            )
        )

        self.assertEqual(vehicles.type_parameters["typeB"][
                             "car_following_params"].speed_mode, 0)
        self.assertEqual(vehicles.type_parameters["typeB"][
                             "lane_change_params"].lane_change_mode, 512)

        vehicles.add(
            "typeC",
            acceleration_controller=(IDMController, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=31,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=277
            )
        )

        self.assertEqual(vehicles.type_parameters["typeC"][
                             "car_following_params"].speed_mode, 31)
        self.assertEqual(vehicles.type_parameters["typeC"][
                             "lane_change_params"].lane_change_mode, 277)

    def test_controlled_id_params(self):
        """
        Ensure that, if a vehicle is not a sumo vehicle, then minGap is set to
        zero so that all headway values are correct.
        """
        # check that, if the vehicle is a SimCarFollowingController vehicle,
        # then its minGap, accel, and decel are set to default
        vehicles = VehicleParams()
        vehicles.add(
            "typeA",
            acceleration_controller=(SimCarFollowingController, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed",
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode="no_lc_safe",
            ))
        default_mingap = SumoCarFollowingParams().controller_params["minGap"]
        self.assertEqual(vehicles.types[0]["type_params"]["minGap"],
                         default_mingap)

    def test_add_vehicles_human(self):
        """
        Ensure that added human vehicles are placed in the current vehicle
        IDs, and that the number of vehicles is correct.
        """
        # generate a vehicles class
        vehicles = VehicleParams()

        # vehicles whose acceleration and LC are controlled by sumo
        vehicles.add("test_1", num_vehicles=1)

        # vehicles whose acceleration are controlled by sumo
        vehicles.add(
            "test_2",
            num_vehicles=2,
            lane_change_controller=(StaticLaneChanger, {}))

        # vehicles whose LC are controlled by sumo
        vehicles.add(
            "test_3",
            num_vehicles=4,
            acceleration_controller=(IDMController, {}))

        env, _, _ = ring_road_exp_setup(vehicles=vehicles)

        self.assertEqual(env.k.vehicle.num_vehicles, 7)
        self.assertEqual(len(env.k.vehicle.get_ids()), 7)
        self.assertEqual(len(env.k.vehicle.get_rl_ids()), 0)
        self.assertEqual(len(env.k.vehicle.get_human_ids()), 7)
        self.assertEqual(len(env.k.vehicle.get_controlled_ids()), 4)
        self.assertEqual(len(env.k.vehicle.get_controlled_lc_ids()), 2)

    def test_add_vehicles_rl(self):
        """
        Ensure that added rl vehicles are placed in the current vehicle IDs,
        and that the number of vehicles is correct.
        """
        vehicles = VehicleParams()
        vehicles.add(
            "test_rl",
            num_vehicles=10,
            acceleration_controller=(RLController, {}))

        env, _, _ = ring_road_exp_setup(vehicles=vehicles)

        self.assertEqual(env.k.vehicle.num_vehicles, 10)
        self.assertEqual(len(env.k.vehicle.get_ids()), 10)
        self.assertEqual(len(env.k.vehicle.get_rl_ids()), 10)
        self.assertEqual(len(env.k.vehicle.get_human_ids()), 0)
        self.assertEqual(len(env.k.vehicle.get_controlled_ids()), 0)
        self.assertEqual(len(env.k.vehicle.get_controlled_lc_ids()), 0)

    def test_remove(self):
        """
        Check that there is no trace of the vehicle ID of the vehicle meant to
        be removed in the vehicles class.
        """
        # generate a vehicles class
        vehicles = VehicleParams()
        vehicles.add("test", num_vehicles=10)
        vehicles.add(
            "test_rl",
            num_vehicles=10,
            acceleration_controller=(RLController, {}))

        env, _, _ = ring_road_exp_setup(vehicles=vehicles)

        # remove one human-driven vehicle and on rl vehicle
        env.k.vehicle.remove("test_0")
        env.k.vehicle.remove("test_rl_0")

        # ensure that the removed vehicle's ID is not in any lists of vehicles
        self.assertTrue("test_0" not in env.k.vehicle.get_ids(),
                        msg="vehicle still in get_ids()")
        self.assertTrue("test_0" not in env.k.vehicle.get_human_ids(),
                        msg="vehicle still in get_controlled_lc_ids()")
        self.assertTrue("test_0" not in env.k.vehicle.get_controlled_lc_ids(),
                        msg="vehicle still in get_controlled_lc_ids()")
        self.assertTrue("test_0" not in env.k.vehicle.get_controlled_ids(),
                        msg="vehicle still in get_controlled_ids()")
        self.assertTrue("test_rl_0" not in env.k.vehicle.get_ids(),
                        msg="RL vehicle still in get_ids()")
        self.assertTrue("test_rl_0" not in env.k.vehicle.get_rl_ids(),
                        msg="RL vehicle still in get_rl_ids()")

        # ensure that the vehicles are not storing extra information in the
        # vehicles.__vehicles dict
        error_state = env.k.vehicle.get_speed('test_0', error=None)
        self.assertIsNone(error_state)
        error_state_rl = env.k.vehicle.get_speed('rl_test_0', error=None)
        self.assertIsNone(error_state_rl)

        # ensure that the num_vehicles matches the actual number of vehicles
        self.assertEqual(env.k.vehicle.num_vehicles,
                         len(env.k.vehicle.get_ids()))

        # ensures that then num_rl_vehicles matches the actual number of rl veh
        self.assertEqual(env.k.vehicle.num_rl_vehicles,
                         len(env.k.vehicle.get_rl_ids()))


class TestMultiLaneData(unittest.TestCase):
    """
    Tests the functions get_lane_leaders(), get_lane_followers(),
    get_lane_headways(), and get_lane_tailways() in the Vehicles class.
    """

    def test_no_junctions_ring(self):
        """
        Test the above mentioned methods in the absence of junctions.
        """
        # setup a network with no junctions and several vehicles
        # also, setup with a deterministic starting position to ensure that the
        # headways/lane leaders are what is expected
        additional_net_params = {
            "length": 230,
            "lanes": 3,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(RLController, {}),
            num_vehicles=21)

        initial_config = InitialConfig(lanes_distribution=float("inf"))

        env, _, _ = ring_road_exp_setup(
            net_params=net_params,
            vehicles=vehicles,
            initial_config=initial_config)
        env.reset()

        # check the lane leaders method is outputting the right values
        actual_lane_leaders = env.k.vehicle.get_lane_leaders("test_0")
        expected_lane_leaders = ["test_3", "test_1", "test_2"]
        self.assertCountEqual(actual_lane_leaders, expected_lane_leaders)

        # check the lane headways is outputting the right values
        actual_lane_head = env.k.vehicle.get_lane_headways("test_0")
        expected_lane_head = [27.85714285714286, -5, -5]
        self.assertCountEqual(actual_lane_head, expected_lane_head)

        # check the lane followers method is outputting the right values
        actual_lane_followers = env.k.vehicle.get_lane_followers("test_0")
        expected_lane_followers = ["test_18", "test_19", "test_20"]
        self.assertCountEqual(actual_lane_followers, expected_lane_followers)

        # check the lane tailways is outputting the right values
        actual_lane_tail = env.k.vehicle.get_lane_tailways("test_0")
        expected_lane_tail = [28.277143] * 3
        np.testing.assert_array_almost_equal(actual_lane_tail,
                                             expected_lane_tail)

    def test_no_junctions_highway(self):
        additional_net_params = {
            "length": 100,
            "lanes": 3,
            "speed_limit": 30,
            "resolution": 40,
            "num_edges": 1,
            "use_ghost_edge": False,
            "ghost_speed_limit": 25,
            "boundary_cell_length": 300,
        }
        net_params = NetParams(additional_params=additional_net_params)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(RLController, {}),
            num_vehicles=3,
            initial_speed=1.0)

        # Test Cases
        # 1. If there's only one vehicle in each lane, we should still
        # find one leader and one follower for the central vehicle
        initial_config = InitialConfig(lanes_distribution=float("inf"))
        initial_config.spacing = "custom"
        initial_pos = {"start_positions": [('highway_0', 20),
                                           ('highway_0', 30),
                                           ('highway_0', 10)],
                       "start_lanes": [1, 2, 0]}
        initial_config.additional_params = initial_pos

        env, _, _ = highway_exp_setup(
            sim_params=SumoParams(sim_step=0.1, render=False),
            net_params=net_params,
            vehicles=vehicles,
            initial_config=initial_config)
        env.reset()

        # test the central car
        # test_0 is car to test in central lane
        # test_1 should be leading car in lane 2
        # test_2 should be trailing car in lane 0
        actual_lane_leaders = env.k.vehicle.get_lane_leaders("test_0")
        expected_lane_leaders = ["", "", "test_1"]
        self.assertTrue(actual_lane_leaders == expected_lane_leaders)
        actual_lane_headways = env.k.vehicle.get_lane_headways("test_0")
        expected_lane_headways = [1000, 1000, 5.0]
        np.testing.assert_array_almost_equal(actual_lane_headways,
                                             expected_lane_headways)

        actual_lane_followers = env.k.vehicle.get_lane_followers("test_0")
        expected_lane_followers = ["test_2", "", ""]
        self.assertTrue(actual_lane_followers == expected_lane_followers)
        actual_lane_tailways = env.k.vehicle.get_lane_tailways("test_0")
        expected_lane_tailways = [5.0, 1000, 1000]
        np.testing.assert_array_almost_equal(actual_lane_tailways,
                                             expected_lane_tailways)

        # test the leader/follower speed methods
        expected_leader_speed = [0.0, 0.0, 1.0]
        actual_leader_speed = env.k.vehicle.get_lane_leaders_speed("test_0")
        np.testing.assert_array_almost_equal(actual_leader_speed,
                                             expected_leader_speed)

        expected_follower_speed = [1.0, 0.0, 0.0]
        actual_follower_speed = env.k.vehicle.get_lane_followers_speed(
            "test_0")
        np.testing.assert_array_almost_equal(actual_follower_speed,
                                             expected_follower_speed)

        # Next, test the case where all vehicles are on the same
        # edge and there's two vehicles in each lane
        # Cases to test
        # 1. For lane 0, should find a leader and follower for tested car
        # 2. For lane 1, both vehicles are behind the test car
        # 3. For lane 2, both vehicles are in front of the tested car
        # 4. For lane 3, one vehicle in front and one behind the tested car
        additional_net_params = {
            "length": 100,
            "lanes": 4,
            "speed_limit": 30,
            "resolution": 40,
            "num_edges": 1,
            "use_ghost_edge": False,
            "ghost_speed_limit": 25,
            "boundary_cell_length": 300,
        }
        net_params = NetParams(additional_params=additional_net_params)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(RLController, {}),
            num_vehicles=9,
            initial_speed=1.0)

        initial_config = InitialConfig(lanes_distribution=float("inf"))
        initial_config.spacing = "custom"
        initial_pos = {"start_positions": [('highway_0', 50),
                                           ('highway_0', 60),
                                           ('highway_0', 40),
                                           ('highway_0', 40),
                                           ('highway_0', 30),
                                           ('highway_0', 60),
                                           ('highway_0', 70),
                                           ('highway_0', 60),
                                           ('highway_0', 40),
                                           ],
                       "start_lanes": [0, 0, 0, 1, 1, 2, 2, 3, 3]}
        initial_config.additional_params = initial_pos

        env, _, _ = highway_exp_setup(
            sim_params=SumoParams(sim_step=0.1, render=False),
            net_params=net_params,
            vehicles=vehicles,
            initial_config=initial_config)
        env.reset()

        actual_lane_leaders = env.k.vehicle.get_lane_leaders("test_0")
        expected_lane_leaders = ["test_1", "", "test_5", "test_7"]
        self.assertTrue(actual_lane_leaders == expected_lane_leaders)

        actual_lane_headways = env.k.vehicle.get_lane_headways("test_0")
        expected_lane_headways = [5.0, 1000, 5.0, 5.0]
        np.testing.assert_array_almost_equal(actual_lane_headways,
                                             expected_lane_headways)

        actual_lane_followers = env.k.vehicle.get_lane_followers("test_0")
        expected_lane_followers = ["test_2", "test_3", "", "test_8"]
        self.assertTrue(actual_lane_followers == expected_lane_followers)

        actual_lane_tailways = env.k.vehicle.get_lane_tailways("test_0")
        expected_lane_tailways = [5.0, 5.0, 1000, 5.0]
        np.testing.assert_array_almost_equal(actual_lane_tailways,
                                             expected_lane_tailways)

        # test the leader/follower speed methods
        expected_leader_speed = [1.0, 0.0, 1.0, 1.0]
        actual_leader_speed = env.k.vehicle.get_lane_leaders_speed("test_0")
        np.testing.assert_array_almost_equal(actual_leader_speed,
                                             expected_leader_speed)
        expected_follower_speed = [1.0, 1.0, 0.0, 1.0]
        actual_follower_speed = env.k.vehicle.get_lane_followers_speed(
            "test_0")
        np.testing.assert_array_almost_equal(actual_follower_speed,
                                             expected_follower_speed)

        # Now test if all the vehicles are on different edges and
        # different lanes
        additional_net_params = {
            "length": 100,
            "lanes": 3,
            "speed_limit": 30,
            "resolution": 40,
            "num_edges": 3,
            "use_ghost_edge": False,
            "ghost_speed_limit": 25,
            "boundary_cell_length": 300,
        }
        net_params = NetParams(additional_params=additional_net_params)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(RLController, {}),
            num_vehicles=3,
            initial_speed=1.0)

        # Test Cases
        # 1. If there's only one vehicle in each lane, we should still
        # find one leader and one follower for the central vehicle
        initial_config = InitialConfig(lanes_distribution=float("inf"))
        initial_config.spacing = "custom"
        initial_pos = {"start_positions": [('highway_1', 50 - (100 / 3.0)),
                                           ('highway_2', 75 - (2 * 100 / 3.0)),
                                           ('highway_0', 25)],
                       "start_lanes": [1, 2, 0]}
        initial_config.additional_params = initial_pos

        env, _, _ = highway_exp_setup(
            sim_params=SumoParams(sim_step=0.1, render=False),
            net_params=net_params,
            vehicles=vehicles,
            initial_config=initial_config)
        env.reset()

        # test the central car
        # test_0 is car to test in central lane
        # test_1 should be leading car in lane 2
        # test_2 should be trailing car in lane 0

        actual_lane_leaders = env.k.vehicle.get_lane_leaders("test_0")
        expected_lane_leaders = ["", "", "test_1"]
        self.assertTrue(actual_lane_leaders == expected_lane_leaders)
        actual_lane_headways = env.k.vehicle.get_lane_headways("test_0")
        expected_lane_headways = [1000, 1000, 22.996667]
        np.testing.assert_array_almost_equal(actual_lane_headways,
                                             expected_lane_headways)

        actual_lane_followers = env.k.vehicle.get_lane_followers("test_0")
        expected_lane_followers = ["test_2", "", ""]
        self.assertTrue(actual_lane_followers == expected_lane_followers)
        actual_lane_tailways = env.k.vehicle.get_lane_tailways("test_0")
        expected_lane_tailways = [20.096667, 1000, 1000]
        np.testing.assert_array_almost_equal(actual_lane_tailways,
                                             expected_lane_tailways)

        # test the leader/follower speed methods
        expected_leader_speed = [0.0, 0.0, 1.0]
        actual_leader_speed = env.k.vehicle.get_lane_leaders_speed("test_0")
        np.testing.assert_array_almost_equal(actual_leader_speed,
                                             expected_leader_speed)
        expected_follower_speed = [1.0, 0.0, 0.0]
        actual_follower_speed = env.k.vehicle.get_lane_followers_speed(
            "test_0")
        np.testing.assert_array_almost_equal(actual_follower_speed,
                                             expected_follower_speed)

        # Now test if all the vehicles are on different edges and same
        # lanes
        additional_net_params = {
            "length": 100,
            "lanes": 3,
            "speed_limit": 30,
            "resolution": 40,
            "num_edges": 3,
            "use_ghost_edge": False,
            "ghost_speed_limit": 25,
            "boundary_cell_length": 300,
        }
        net_params = NetParams(additional_params=additional_net_params)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(RLController, {}),
            num_vehicles=3,
            initial_speed=1.0)

        # Test Cases
        # 1. If there's only one vehicle in each lane, we should still
        # find one leader and one follower for the central vehicle
        initial_config = InitialConfig(lanes_distribution=float("inf"))
        initial_config.spacing = "custom"
        initial_pos = {"start_positions": [('highway_1', 50 - (100 / 3.0)),
                                           ('highway_2', 75 - (2 * 100 / 3.0)),
                                           ('highway_0', 25)],
                       "start_lanes": [0, 0, 0]}
        initial_config.additional_params = initial_pos

        env, _, _ = highway_exp_setup(
            sim_params=SumoParams(sim_step=0.1, render=False),
            net_params=net_params,
            vehicles=vehicles,
            initial_config=initial_config)
        env.reset()

        # test the central car
        # test_0 is car to test in lane 0
        # test_1 should be leading car in lane 0
        # test_2 should be trailing car in lane 0
        actual_lane_leaders = env.k.vehicle.get_lane_leaders("test_0")
        expected_lane_leaders = ["test_1", "", ""]
        self.assertTrue(actual_lane_leaders == expected_lane_leaders)
        actual_lane_headways = env.k.vehicle.get_lane_headways("test_0")
        expected_lane_headways = [22.996667, 1000, 1000]
        np.testing.assert_array_almost_equal(actual_lane_headways,
                                             expected_lane_headways)

        actual_lane_followers = env.k.vehicle.get_lane_followers("test_0")
        expected_lane_followers = ["test_2", "", ""]
        self.assertTrue(actual_lane_followers == expected_lane_followers)
        actual_lane_tailways = env.k.vehicle.get_lane_tailways("test_0")
        expected_lane_tailways = [20.096667, 1000, 1000]
        np.testing.assert_array_almost_equal(actual_lane_tailways,
                                             expected_lane_tailways)

        # test the leader/follower speed methods
        expected_leader_speed = [1.0, 0.0, 0.0]
        actual_leader_speed = env.k.vehicle.get_lane_leaders_speed("test_0")
        np.testing.assert_array_almost_equal(actual_leader_speed,
                                             expected_leader_speed)
        expected_follower_speed = [1.0, 0.0, 0.0]
        actual_follower_speed = env.k.vehicle.get_lane_followers_speed(
            "test_0")
        np.testing.assert_array_almost_equal(actual_follower_speed,
                                             expected_follower_speed)

    def test_junctions(self):
        """
        Test the above mentioned methods in the presence of junctions.
        """
        # TODO(ak): add test
        pass


class TestIdsByEdge(unittest.TestCase):
    """
    Tests the ids_by_edge() method
    """

    def setUp(self):
        # create the environment and network classes for a figure eight
        vehicles = VehicleParams()
        vehicles.add(veh_id="test", num_vehicles=20)

        self.env, _, _ = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # free data used by the class
        self.env.terminate()
        self.env = None

    def test_ids_by_edge(self):
        self.env.reset()
        ids = self.env.k.vehicle.get_ids_by_edge("bottom")
        expected_ids = ["test_0", "test_1", "test_2", "test_3", "test_4"]
        self.assertCountEqual(ids, expected_ids)


class TestObservedIDs(unittest.TestCase):
    """Tests the observed_ids methods, which are used for visualization."""

    def test_obs_ids(self):
        vehicles = VehicleParams()
        vehicles.add(veh_id="test", num_vehicles=10)

        env, _, _ = ring_road_exp_setup(vehicles=vehicles)

        # test setting new observed values
        env.k.vehicle.set_observed("test_0")
        self.assertCountEqual(env.k.vehicle.get_observed_ids(), ["test_0"])

        env.k.vehicle.set_observed("test_1")
        self.assertCountEqual(env.k.vehicle.get_observed_ids(),
                              ["test_0", "test_1"])

        # ensures that setting vehicles twice doesn't add an element
        env.k.vehicle.set_observed("test_0")
        self.assertListEqual(env.k.vehicle.get_observed_ids(),
                             ["test_0", "test_1"])

        # test removing observed values
        env.k.vehicle.remove_observed("test_0")
        self.assertCountEqual(env.k.vehicle.get_observed_ids(), ["test_1"])

        # ensures that removing a value that does not exist does not lead to
        # an error
        env.k.vehicle.remove_observed("test_0")
        self.assertCountEqual(env.k.vehicle.get_observed_ids(), ["test_1"])


if __name__ == '__main__':
    unittest.main()
