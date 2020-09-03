"""Contains the figure eight network class."""

import numpy as np
from numpy import pi, sin, cos, linspace

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.networks.base import Network

ADDITIONAL_NET_PARAMS = {
    # radius of the circular components
    "radius_ring": 30,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curved portions
    "resolution": 40
}


class FigureEightNetwork(Network):
    """Figure eight network class.

    The figure eight network is an extension of the ring road network: Two
    rings, placed at opposite ends of the network, are connected by an
    intersection with road segments of length equal to the diameter of the
    rings. Serves as a simulation of a closed ring intersection.

    Requires from net_params:

    * **ring_radius** : radius of the circular portions of the network. Also
      corresponds to half the length of the perpendicular straight lanes.
    * **resolution** : number of nodes resolution in the circular portions
    * **lanes** : number of lanes in the network
    * **speed** : max speed of vehicles in the network

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import FigureEightNetwork
    >>>
    >>> network = FigureEightNetwork(
    >>>     name='figure_eight',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'radius_ring': 50,
    >>>             'lanes': 75,
    >>>             'speed_limit': 30,
    >>>             'resolution': 40
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a figure 8 network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        ring_radius = net_params.additional_params["radius_ring"]
        self.ring_edgelen = ring_radius * np.pi / 2.
        self.intersection_len = 2 * ring_radius

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]

        nodes = [{
            "id": "center",
            "x": 0,
            "y": 0,
            "radius": (2.9 + 3.3 * net_params.additional_params["lanes"])/2,
            "type": "priority"
        }, {
            "id": "right",
            "x": r,
            "y": 0,
            "type": "priority"
        }, {
            "id": "top",
            "x": 0,
            "y": r,
            "type": "priority"
        }, {
            "id": "left",
            "x": -r,
            "y": 0,
            "type": "priority"
        }, {
            "id": "bottom",
            "x": 0,
            "y": -r,
            "type": "priority"
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]
        resolution = net_params.additional_params["resolution"]
        ring_edgelen = 3 * r * pi / 2.
        intersection_edgelen = 2 * r

        # intersection edges
        edges = [{
            "id": "bottom",
            "type": "edgeType",
            "priority": "78",
            "from": "bottom",
            "to": "center",
            "length": intersection_edgelen / 2
        }, {
            "id": "top",
            "type": "edgeType",
            "priority": 78,
            "from": "center",
            "to": "top",
            "length": intersection_edgelen / 2
        }, {
            "id": "right",
            "type": "edgeType",
            "priority": 46,
            "from": "right",
            "to": "center",
            "length": intersection_edgelen / 2
        }, {
            "id": "left",
            "type": "edgeType",
            "priority": 46,
            "from": "center",
            "to": "left",
            "length": intersection_edgelen / 2
        }]

        # ring edges
        edges += [{
            "id": "upper_ring",
            "type": "edgeType",
            "from": "top",
            "to": "right",
            "length": ring_edgelen,
            "shape": [(r * (1 - cos(t)), r * (1 + sin(t)))
                      for t in linspace(0, 3 * pi / 2, resolution)]
        }, {
            "id": "lower_ring",
            "type": "edgeType",
            "from": "left",
            "to": "bottom",
            "length": ring_edgelen,
            "shape": [(-r + r * cos(t), -r + r * sin(t))
                      for t in linspace(pi / 2, 2 * pi, resolution)]
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        types = [{
            "id": "edgeType",
            "numLanes": lanes,
            "speed": speed_limit
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "bottom":
                ["bottom", "top", "upper_ring", "right", "left", "lower_ring"],
            "top":
                ["top", "upper_ring", "right", "left", "lower_ring", "bottom"],
            "upper_ring":
                ["upper_ring", "right", "left", "lower_ring", "bottom", "top"],
            "left":
                ["left", "lower_ring", "bottom", "top", "upper_ring", "right"],
            "right":
                ["right", "left", "lower_ring", "bottom", "top", "upper_ring"],
            "lower_ring":
                ["lower_ring", "bottom", "top", "upper_ring", "right", "left"],
        }

        return rts

    def specify_connections(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        conn_dict = {}
        conn = []
        for i in range(lanes):
            conn += [{"from": "bottom",
                      "to": "top",
                      "fromLane": str(i),
                      "toLane": str(i)}]
            conn += [{"from": "right",
                      "to": "left",
                      "fromLane": str(i),
                      "toLane": str(i)}]
        conn_dict["center"] = conn
        return conn_dict

    def specify_edge_starts(self):
        """See base class."""
        edgestarts = [
            ("bottom", 0),
            ("top", self.intersection_len / 2),
            ("upper_ring", self.intersection_len),
            ("right", self.intersection_len + 3 * self.ring_edgelen),
            ("left", 3 / 2 * self.intersection_len + 3 * self.ring_edgelen),
            ("lower_ring", 2 * self.intersection_len + 3 * self.ring_edgelen)]

        return edgestarts
