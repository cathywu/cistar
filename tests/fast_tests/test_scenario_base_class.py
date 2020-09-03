import unittest
import os
import numpy as np

from flow.config import PROJECT_PATH
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import VehicleParams
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import TestEnv
from flow.networks import Network

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController

from tests.setup_scripts import ring_road_exp_setup, figure_eight_exp_setup, \
    highway_exp_setup
from tests.setup_scripts import variable_lanes_exp_setup

os.environ["TEST_FLAG"] = "True"

JUNCTION_LENGTH = 0.6


class NoRouteNetwork(RingNetwork):
    """A network with no routes.

    Used to check for default route assignment.
    """

    def specify_routes(self, net_params):
        return None


class TestGetX(unittest.TestCase):
    """
    Tests the get_x function for vehicles placed in links and in junctions.
    This is tested on a network whose edgestarts are known beforehand
    (figure 8).
    """

    def setUp(self):
        # create the environment and network classes for a figure eight
        self.env, _, _ = figure_eight_exp_setup()

    def tearDown(self):
        # free data used by the class
        self.env = None

    def test_getx(self):
        # test for an edge in the lanes
        edge_1 = "bottom"
        pos_1 = 4.72
        self.assertAlmostEqual(self.env.k.network.get_x(edge_1, pos_1), 4.72)

        # test for an edge in the internal links
        edge_2 = ":top_0"
        pos_2 = 0.1
        self.assertAlmostEqual(self.env.k.network.get_x(edge_2, pos_2), 69.5)

    def test_error(self):
        edge = ''
        pos = 4.72
        self.assertAlmostEqual(self.env.k.network.get_x(edge, pos), -1001)


class TestGetEdge(unittest.TestCase):
    """
    Tests the get_edge function for vehicles placed in links and in internal
    edges. This is tested on a network whose edgestarts are known beforehand
    (figure 8).
    """

    def setUp(self):
        # create the environment and network classes for a figure eight
        self.env, _, _ = figure_eight_exp_setup()

    def tearDown(self):
        # free data used by the class
        self.env.terminate()
        self.env = None

    def test_get_edge(self):
        # test for a position in the lanes
        x1 = 5
        self.assertTupleEqual(
            self.env.k.network._get_edge(x1), ("bottom", 5))


class TestEvenStartPos(unittest.TestCase):
    """
    Tests the function gen_even_start_pos in networks/base.py. This function
    can be used on any network subclass, and therefore may be tested on any of
    these classes. In order to perform this testing, replace the network in
    setUp() with the network to be tested.
    """

    def setUp_gen_start_pos(self, initial_config=InitialConfig()):
        """
        Replace with any network you would like to test gen_even_start_pos on.
        In ordering for all the tests to be meaningful, the network must
        contain MORE THAN TWO LANES.
        """
        # create a multi-lane ring road network
        additional_net_params = {
            "length": 230,
            "lanes": 4,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # place 15 vehicles in the network (we need at least more than 1)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=0
            ),
            num_vehicles=15)

        # create the environment and network classes for a ring road
        self.env, _, _ = ring_road_exp_setup(
            net_params=net_params,
            initial_config=initial_config,
            vehicles=vehicles)

    def tearDown_gen_start_pos(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_base(self):
        """
        Tests that get_even_start_pos function evenly distributed vehicles in a
        network.
        """
        # set the initial_config parameters as default (even spacing, no extra
        # conditions), and reset
        initial_config = InitialConfig(lanes_distribution=1)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.k.vehicle.get_ids()
        veh_pos = np.array([self.env.k.vehicle.get_x_by_id(veh_id)
                            for veh_id in ids])

        # to account for the additions to the positions by the junctions
        veh_pos[veh_pos > (230/4)] -= JUNCTION_LENGTH
        veh_pos[veh_pos > (230/2)] -= JUNCTION_LENGTH
        veh_pos[veh_pos > (230*3/4)] -= JUNCTION_LENGTH

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(
            np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
            self.env.k.network.non_internal_length())

        # check that the position of the first vehicle is at 0
        self.assertEqual(veh_pos[0], 0)

        # if all element are equal, there should only be one unique value
        self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_x0(self):
        """
        Tests that the vehicles are uniformly distributed and the initial
        vehicle if at a given x0.
        """
        # set the initial_config parameters with an x0 value that is something
        # in between zero and the length of the network
        x0 = 10
        initial_config = InitialConfig(x0=x0, lanes_distribution=1)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.k.vehicle.get_ids()
        veh_pos = np.array([self.env.k.vehicle.get_x_by_id(veh_id)
                            for veh_id in ids])

        # to account for the additions to the positions by the junctions
        veh_pos[veh_pos > (230/4)] -= JUNCTION_LENGTH
        veh_pos[veh_pos > (230/2)] -= JUNCTION_LENGTH
        veh_pos[veh_pos > (230*3/4)] -= JUNCTION_LENGTH

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(
            np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
            self.env.k.network.non_internal_length())

        # check that the position of the first vehicle is at 0
        self.assertEqual(veh_pos[0], x0 % self.env.k.network.non_internal_length())

        # if all element are equal, there should only be one unique value
        self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_bunching(self):
        """
        Tests that vehicles are uniformly distributed given a certain bunching
        """
        # set the initial_config parameters with a modest bunching term
        bunching = 10
        initial_config = InitialConfig(bunching=bunching, lanes_distribution=1)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.k.vehicle.get_ids()
        veh_pos = np.array([self.env.k.vehicle.get_x_by_id(veh_id)
                            for veh_id in ids])

        # to account for the additions to the positions by the junctions
        veh_pos[veh_pos > (230/4)] -= JUNCTION_LENGTH
        veh_pos[veh_pos > (230/2)] -= JUNCTION_LENGTH
        veh_pos[veh_pos > (230*3/4)] -= JUNCTION_LENGTH

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(
            np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
            self.env.k.network.non_internal_length())

        # check that all vehicles except the last vehicle have the same spacing
        self.assertEqual(np.unique(np.around(nth_headway[:-1], 2)).size, 1)

        # check that the spacing of the last vehicle is just offset by the
        # bunching term
        self.assertAlmostEqual(nth_headway[-1] - nth_headway[-2], bunching)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_bunching_too_small(self):
        """
        Tests that if bunching is negative, it is set to 0
        """
        # set the initial_config parameters with a negative bunching term
        bunching = -10
        initial_config = InitialConfig(bunching=bunching, lanes_distribution=1)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        startpos, _ = self.env.k.network.generate_starting_positions(
            initial_config=initial_config
        )

        # get the positions of all vehicles
        veh_pos = np.array([self.env.k.network.get_x(pos[0], pos[1])
                            for pos in startpos])

        # to account for the additions to the positions by the junctions
        veh_pos[veh_pos > (230/4)] -= JUNCTION_LENGTH
        veh_pos[veh_pos > (230/2)] -= JUNCTION_LENGTH
        veh_pos[veh_pos > (230*3/4)] -= JUNCTION_LENGTH

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(
            np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
            self.env.k.network.non_internal_length())

        # check that all vehicles, including the last vehicle, have the same
        # spacing
        self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_lanes_distribution(self):
        """
        Tests that if lanes_distribution is less than the total number of
        lanes, the vehicles are uniformly distributed over the specified number
        of lanes.
        """
        # set the initial_config parameters with a lanes distribution less than
        # the number of lanes in the network.
        lanes_distribution = 2
        initial_config = InitialConfig(lanes_distribution=lanes_distribution)

        # create the environment
        self.setUp_gen_start_pos(initial_config)
        lanes = self.env.net_params.additional_params['lanes']

        # get the positions of all vehicles
        ids = self.env.k.vehicle.get_ids()
        veh_pos = []
        for i in range(lanes):
            veh_pos.append(np.array([
                self.env.k.vehicle.get_x_by_id(veh_id) for veh_id in ids
                if self.env.k.vehicle.get_lane(veh_id) == i
            ]))

            # to account for the additions to the positions by the junctions
            veh_pos[-1][veh_pos[-1] > (230 / 4)] -= JUNCTION_LENGTH
            veh_pos[-1][veh_pos[-1] > (230 / 2)] -= JUNCTION_LENGTH
            veh_pos[-1][veh_pos[-1] > (230 * 3 / 4)] -= JUNCTION_LENGTH

        # check that the vehicles are uniformly distributed in the number of
        # requested lanes
        for i in range(lanes_distribution):
            # difference in position between the nth vehicle and the vehicle
            # ahead of it
            nth_headway = \
                np.mod(np.append(veh_pos[i][1:], veh_pos[i][0]) - veh_pos[i],
                       self.env.k.network.non_internal_length())

            self.assertEqual(np.unique(np.around(nth_headway[:-1], 2)).size, 1)

        # check that there are no vehicles in the remaining lanes
        for i in range(lanes - lanes_distribution):
            self.assertEqual(len(veh_pos[i + lanes_distribution]), 0)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_lanes_distribution_too_small(self):
        """
        Tests that when lanes_distribution is less than 1, the number is set
        to 1.
        """
        # set the initial_config parameters with a small or negative
        # lanes_distribution
        lanes_distribution = np.random.randint(-100, 0)
        initial_config = InitialConfig(lanes_distribution=lanes_distribution)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # check that only the first lane has vehicles
        ids = self.env.k.vehicle.get_ids()
        veh_lanes = [self.env.k.vehicle.get_lane(veh_id) for veh_id in ids]
        self.assertEqual(np.unique(veh_lanes).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_lanes_distribution_too_large(self):
        """
        Tests that when lanes_distribution is greater than the number of lanes,
        the vehicles are distributed over the maximum number of lanes instead.
        """
        # set the initial_config parameter with a very large lanes_distribution
        lanes_distribution = np.inf
        initial_config = InitialConfig(lanes_distribution=lanes_distribution)

        # create the environment
        self.setUp_gen_start_pos(initial_config)
        lanes = self.env.net_params.additional_params['lanes']

        # get the positions of all vehicles
        ids = self.env.k.vehicle.get_ids()
        veh_pos = []
        for i in range(lanes):
            veh_pos.append([
                self.env.k.vehicle.get_x_by_id(veh_id) for veh_id in ids
                if self.env.k.vehicle.get_lane(veh_id) == i
            ])

        # check that the vehicles are uniformly distributed in the number of
        # requested lanes lanes
        for i in range(lanes):
            # difference in position between the nth vehicle and the vehicle
            # ahead of it
            nth_headway = \
                np.mod(np.append(veh_pos[i][1:], veh_pos[i][0]) - veh_pos[i],
                       self.env.k.network.non_internal_length())

            self.assertEqual(np.unique(np.around(nth_headway[:-1], 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_edges_distribution(self):
        """
        Tests that vehicles are only placed in edges listed in the
        edges_distribution parameter, when edges are specified
        """
        # set the initial_config parameters with an edges_distribution term for
        # only a few edges
        edges = ["top", "bottom"]
        initial_config = InitialConfig(edges_distribution=edges)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # check that all vehicles are only placed in edges specified in the
        # edges_distribution term
        for veh_id in self.env.k.vehicle.get_ids():
            self.assertTrue(self.env.k.vehicle.get_edge(veh_id) in edges)

    def test_edges_distribution_dict(self):
        """
        Tests that vehicles of the correct quantity are placed on each edge
        when edges_distribution is a dict.
        """
        # test that when the number of vehicles don't match an AssertionError
        # is raised
        edges = {"top": 2, "bottom": 1}
        initial_config = InitialConfig(edges_distribution=edges)
        self.assertRaises(AssertionError, self.setUp_gen_start_pos,
                          initial_config=initial_config)

        # verify that the correct number of vehicles are placed in each edge
        edges = {"top": 5, "bottom": 6, "left": 4}
        initial_config = InitialConfig(edges_distribution=edges)
        self.setUp_gen_start_pos(initial_config)

        for edge in edges:
            self.assertEqual(len(self.env.k.vehicle.get_ids_by_edge(edge)),
                             edges[edge])

    def test_num_vehicles(self):
        """
        Tests that the number of starting positions generated is:
        - the number of vehicles in the vehicles class is no "num_vehicles"
          parameter is specified
        - equal to "num_vehicles" if this value is specified
        """
        # create the environment
        self.setUp_gen_start_pos()

        # check when "num_vehicles" is not specified
        pos, lanes = self.env.k.network.generate_starting_positions(
            initial_config=InitialConfig())
        self.assertEqual(len(pos), self.env.k.vehicle.num_vehicles)
        self.assertEqual(len(lanes), self.env.k.vehicle.num_vehicles)

        # check when "num_vehicles" is specified
        pos, lanes = self.env.k.network.generate_starting_positions(
            initial_config=InitialConfig(), num_vehicles=10)
        self.assertEqual(len(pos), 10)
        self.assertEqual(len(lanes), 10)


class TestRandomStartPos(unittest.TestCase):
    """
    Tests the function gen_random_start_pos in networks/base.py.
    """

    def setUp_gen_start_pos(self, initial_config=InitialConfig()):
        # ensures that the random starting position method is being used
        initial_config.spacing = "random"

        # create a multi-lane ring road network
        additional_net_params = {
            "length": 230,
            "lanes": 4,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=0
            ),
            num_vehicles=5)

        # create the environment and network classes for a ring road
        self.env, _, _ = ring_road_exp_setup(
            net_params=net_params,
            initial_config=initial_config,
            vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_lanes_distribution(self):
        """
        Tests that vehicles are only placed in the requested number of lanes.
        """
        # create the environment
        initial_config = InitialConfig(spacing="random", lanes_distribution=2)
        self.setUp_gen_start_pos(initial_config)

        # verify that all vehicles are located in the number of allocated lanes
        for veh_id in self.env.k.vehicle.get_ids():
            self.assertLess(self.env.k.vehicle.get_lane(veh_id),
                            initial_config.lanes_distribution)

    def test_edges_distribution(self):
        """
        Tests that vehicles are only placed in the requested edges.
        """
        # set the initial_config parameters with an edges_distribution term for
        # only a few edges
        edges = ["top", "bottom"]
        initial_config = InitialConfig(
            spacing="random", edges_distribution=edges)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # check that all vehicles are only placed in edges specified in the
        # edges_distribution term
        for veh_id in self.env.k.vehicle.get_ids():
            self.assertTrue(self.env.k.vehicle.get_edge(veh_id) in edges)

    def test_edges_distribution_dict(self):
        """
        Tests that vehicles of the correct quantity are placed on each edge
        when edges_distribution is a dict.
        """
        # test that when the number of vehicles don't match an AssertionError
        # is raised
        edges = {"top": 2, "bottom": 1}
        initial_config = InitialConfig(edges_distribution=edges)
        self.assertRaises(AssertionError, self.setUp_gen_start_pos,
                          initial_config=initial_config)

        # verify that the correct number of vehicles are placed in each edge
        edges = {"top": 2, "bottom": 3, "left": 0}
        initial_config = InitialConfig(edges_distribution=edges)
        self.setUp_gen_start_pos(initial_config)

        for edge in edges:
            self.assertEqual(len(self.env.k.vehicle.get_ids_by_edge(edge)),
                             edges[edge])


class TestEvenStartPosVariableLanes(unittest.TestCase):
    def setUp(self):
        # place 15 vehicles in the network (we need at least more than 1)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=0
            ),
            num_vehicles=50)

        initial_config = InitialConfig(lanes_distribution=5)

        # create the environment and network classes for a variable lanes per
        # edge ring road
        self.env, _, _ = variable_lanes_exp_setup(
            vehicles=vehicles, initial_config=initial_config)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_even_start_pos_coverage(self):
        """
        Ensure that the vehicles cover all possible lanes.
        """
        lanes = self.env.k.vehicle.get_lane(self.env.k.vehicle.get_ids())
        self.assertFalse(any(i not in lanes for i in range(4)))


class TestRandomStartPosVariableLanes(TestEvenStartPosVariableLanes):
    def setUp(self):
        # place 15 vehicles in the network (we need at least more than 1)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=0
            ),
            num_vehicles=50)

        initial_config = InitialConfig(spacing="random", lanes_distribution=5)

        # create the environment and network classes for a variable lanes per
        # edge ring road
        self.env, _, _ = variable_lanes_exp_setup(
            vehicles=vehicles, initial_config=initial_config)


class TestEdgeLength(unittest.TestCase):
    """
    Tests the edge_length() method in the base network class.
    """

    def test_edge_length_edges(self):
        """
        Tests the edge_length() method when called on edges
        """
        additional_net_params = {
            "length": 1000,
            "lanes": 2,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and network classes for a ring road
        env, _, _ = ring_road_exp_setup(net_params=net_params)

        self.assertEqual(env.k.network.edge_length("top"), 250)

        # test for errors as well
        self.assertAlmostEqual(env.k.network.edge_length("wrong_name"), -1001)

    def test_edge_length_junctions(self):
        """
        Tests the speed_limit() method when called on junctions
        """
        additional_net_params = {
            "radius_ring": 30,
            "lanes": 1,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        env, _, _ = figure_eight_exp_setup(net_params=net_params)

        self.assertAlmostEqual(
            env.k.network.edge_length(":center_0"), 9.40)  # FIXME: 6.2?
        self.assertAlmostEqual(
            env.k.network.edge_length(":center_1"), 9.40)  # FIXME: 6.2?


class TestSpeedLimit(unittest.TestCase):
    """
    Tests the speed_limit() method in the base network class.
    """

    def test_speed_limit_edges(self):
        """
        Tests the speed_limit() method when called on edges
        """
        additional_net_params = {
            "length": 230,
            "lanes": 2,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and network classes for a figure eight
        env, _, _ = ring_road_exp_setup(net_params=net_params)

        self.assertAlmostEqual(env.k.network.speed_limit("top"), 60)

        # test for errors as well
        self.assertAlmostEqual(env.k.network.speed_limit("wrong_name"), -1001)

    def test_speed_limit_junctions(self):
        """
        Tests the speed_limit() method when called on junctions
        """
        additional_net_params = {
            "radius_ring": 30,
            "lanes": 1,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        env, network, _ = figure_eight_exp_setup(net_params=net_params)

        self.assertAlmostEqual(
            env.k.network.speed_limit("bottom"), 60)
        self.assertAlmostEqual(
            env.k.network.speed_limit(":top_0"), 60)


class TestNumLanes(unittest.TestCase):
    """
    Tests the num_lanes() method in the base network class.
    """

    def test_num_lanes_edges(self):
        """
        Tests the num_lanes() method when called on edges
        """
        additional_net_params = {
            "length": 230,
            "lanes": 2,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and network classes for a figure eight
        env, network, _ = ring_road_exp_setup(net_params=net_params)

        self.assertEqual(env.k.network.num_lanes("top"), 2)

        # test for errors as well
        self.assertAlmostEqual(env.k.network.num_lanes("wrong_name"), -1001)

    def test_num_lanes_junctions(self):
        """
        Tests the num_lanes() method when called on junctions
        """
        additional_net_params = {
            "radius_ring": 30,
            "lanes": 3,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        env, network, _ = figure_eight_exp_setup(net_params=net_params)

        self.assertEqual(env.k.network.num_lanes("bottom"), 3)
        self.assertEqual(env.k.network.num_lanes(":top_0"), 3)


class TestGetEdgeList(unittest.TestCase):
    """
    Tests that the get_edge_list() in the network class properly returns all
    edges, and not junctions.
    """

    def setUp(self):
        # create the environment and network classes for a figure eight
        self.env, _, _ = figure_eight_exp_setup()

    def tearDown(self):
        # free data used by the class
        self.env.terminate()
        self.env = None

    def test_get_edge_list(self):
        edge_list = self.env.k.network.get_edge_list()
        expected_edge_list = [
            "bottom", "top", "upper_ring", "right", "left", "lower_ring"]

        self.assertCountEqual(edge_list, expected_edge_list)


class TestGetJunctionList(unittest.TestCase):
    """
    Tests that the get_junction_list() in the network class properly returns
    all junctions, and no edges.
    """

    def setUp(self):
        # create the environment and network classes for a figure eight
        self.env, _, _ = figure_eight_exp_setup()

    def tearDown(self):
        # free data used by the class
        self.env.terminate()
        self.env = None

    def test_get_junction_list(self):
        junction_list = self.env.k.network.get_junction_list()
        expected_junction_list = \
            [':right_0', ':left_0', ':bottom_0', ':top_0', ':center_1',
             ':center_0']

        self.assertCountEqual(junction_list, expected_junction_list)


class TestNextPrevEdge(unittest.TestCase):
    """
    Tests that the next_edge() and prev_edge() methods returns the correct list
    of edges/lanes when looking to a network. This also tests that junctions
    are provided as next edges if they are before the next edge (e.g. a via to
    the next edge)
    """

    def test_next_prev_edge_figure_eight(self):
        """
        Tests the next_edge() and prev_edge() methods for the figure eight.
        """
        env, _, _ = figure_eight_exp_setup()

        next_edge = env.k.network.next_edge("bottom", 0)
        expected_next_edge = [(':center_1', 0)]
        self.assertCountEqual(next_edge, expected_next_edge)

        prev_edge = env.k.network.prev_edge("bottom", 0)
        expected_prev_edge = [(':bottom_0', 0)]
        self.assertCountEqual(prev_edge, expected_prev_edge)

    def test_next_prev_edge_ring_road(self):
        """
        Tests the next_edge() and prev_edge() methods for the ring road.
        """
        env, _, _ = ring_road_exp_setup()

        next_edge = env.k.network.next_edge("top", 0)
        expected_next_edge = [(":left_0", 0)]
        self.assertCountEqual(next_edge, expected_next_edge)

        prev_edge = env.k.network.prev_edge("top", 0)
        expected_prev_edge = [(":top_0", 0)]
        self.assertCountEqual(prev_edge, expected_prev_edge)

    def test_no_edge_ahead(self):
        """
        Tests that, when there are no edges in front, next_edge() returns an
        empty list
        """
        env, _, _ = highway_exp_setup()
        next_edge = env.k.network.next_edge(
            env.k.network.get_edge_list()[0], 0)
        self.assertTrue(len(next_edge) == 0)

    def test_no_edge_behind(self):
        """
        Tests that, when there are no edges behind, prev_edge() returns an
        empty list
        """
        env, _, _ = highway_exp_setup()
        prev_edge = env.k.network.prev_edge(
            env.k.network.get_edge_list()[0], 0)
        self.assertTrue(len(prev_edge) == 0)


class TestDefaultRoutes(unittest.TestCase):

    def test_default_routes(self):
        env_params = EnvParams()
        sim_params = SumoParams(render=False)
        initial_config = InitialConfig()
        vehicles = VehicleParams()
        vehicles.add('human', num_vehicles=1)
        net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

        # create the network
        network = NoRouteNetwork(
            name='bay_bridge',
            net_params=net_params,
            initial_config=initial_config,
            vehicles=vehicles
        )

        # create the environment
        env = TestEnv(
            env_params=env_params,
            sim_params=sim_params,
            network=network
        )

        # check the routes
        self.assertDictEqual(
            env.k.network.rts,
            {"top": [(["top"], 1)],
             "bottom": [(["bottom"], 1)],
             "left": [(["left"], 1)],
             "right": [(["right"], 1)]}
        )


class TestOpenStreetMap(unittest.TestCase):
    """Tests the formation of osm files with Flow. This is done on a section of
    Northside UC Berkeley."""

    def test_sumo(self):
        sim_params = SumoParams()
        vehicles = VehicleParams()
        vehicles.add(veh_id="test")
        env_params = EnvParams()
        net_params = NetParams(
            osm_path=os.path.join(PROJECT_PATH, 'tests/data/euclid.osm'))

        network = Network(
            name="UC-Berkeley-Northside",
            vehicles=vehicles,
            net_params=net_params)

        env = TestEnv(env_params, sim_params, network)

        # check that all the edges were generated
        self.assertEqual(len(env.k.network.get_edge_list()), 29)


class TestNetworkTemplateGenerator(unittest.TestCase):

    def test_network_template(self):
        """Test generate data from network templates.

        This methods tests that routes, vehicle types, and network parameters
        generated from sumo network templates match the expected values. This
        is done on a variant of the figure eight network.
        """
        # generate the network parameters for the figure eight net.xml,
        # rou.xml, and add.xml files
        dir_path = os.path.dirname(os.path.realpath(__file__))
        net_params = NetParams(
            template={
                # network geometry features
                "net": os.path.join(dir_path, "test_files/fig8_test.net.xml"),
                # features associated with the routes vehicles take
                "rou": os.path.join(dir_path, "test_files/fig8_test.rou.xml"),
                # features associated with the properties of drivers
                "vtype": os.path.join(dir_path, "test_files/fig8_test.add.xml")
            },
        )

        # create the network object from the network template files
        network = Network(
            name="template",
            net_params=net_params,
            vehicles=VehicleParams()
        )

        expected_routes = {
            'routetop':
                ['top', 'upper_ring', 'right', 'left', 'lower_ring', 'bottom'],
            'routeupper_ring':
                ['upper_ring', 'right', 'left', 'lower_ring', 'bottom', 'top'],
            'routeleft':
                ['left', 'lower_ring', 'bottom', 'top', 'upper_ring', 'right'],
            'routebottom':
                ['bottom', 'top', 'upper_ring', 'right', 'left', 'lower_ring'],
            'routeright':
                ['right', 'left', 'lower_ring', 'bottom', 'top', 'upper_ring'],
            'routelower_ring':
                ['lower_ring', 'bottom', 'top', 'upper_ring', 'right', 'left']
        }

        expected_cf_params = {
            'controller_params': {
                'speedFactor': 1.0,
                'speedDev': 0.1,
                'carFollowModel': 'IDM',
                'decel': 1.5,
                'impatience': 0.5,
                'maxSpeed': 30.0,
                'accel': 1.0,
                'sigma': 0.5,
                'tau': 1.0,
                'minGap': 0.0
            },
            'speed_mode': 31
        }

        expected_lc_params = {
            'controller_params': {
                'lcCooperative': '1.0',
                'lcKeepRight': '1.0',
                'laneChangeModel': 'LC2013',
                'lcStrategic': '1.0',
                'lcSpeedGain': '1.0'
            },
            'lane_change_mode': 1621
        }

        # test the validity of the outputted results
        self.assertDictEqual(network.routes, expected_routes)
        self.assertDictEqual(network.vehicles.type_parameters['idm']
                             ['car_following_params'].__dict__,
                             expected_cf_params)
        self.assertDictEqual(network.vehicles.type_parameters['idm']
                             ['lane_change_params'].__dict__,
                             expected_lc_params)

        # test for the case of vehicles in rou.xml
        net_params = NetParams(
            template={
                # network geometry features
                "net": os.path.join(dir_path, "test_files/fig8_test.net.xml"),
                # features associated with the routes vehicles take
                "rou": os.path.join(dir_path, "test_files/lust_test.rou.xml"),
                # features associated with the properties of drivers
                "vtype": os.path.join(dir_path, "test_files/fig8_test.add.xml")
            },
        )

        network = Network(
            name="template",
            net_params=net_params,
            vehicles=VehicleParams()
        )

        expected_routes = {
            'h21652c2:1': [
                '--31878#3', '--31878#2', '--31878#1', '--31878#0', '-30872#0',
                '-30872#1', '-30872#2', '-30872#3', '-32750#2', '-32750#3',
                '-32750#4', '-32750#5', '-32750#6', '-32750#7', '-32750#8',
                '-32750#9', '-32750#10', '-32750#11', '-32750#12',
                '--30528#4', '--30528#3', '--30528#2', '--30528#1',
                '--30528#0', '-31492#2', '--32674#9', '--32674#8',
                '--32674#7', '--32674#6', '--32674#5', '--32674#4'
            ]
        }

        self.assertDictEqual(network.routes, expected_routes)


if __name__ == '__main__':
    unittest.main()
