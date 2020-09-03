"""Contains the ring road scenario class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos, linspace

ADDITIONAL_NET_PARAMS = {
    # length of the ring road
    "length": 230,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curves on the ring
    "resolution": 40
}


class RingNetwork(Network):
    """Ring road network.

    This network consists of nodes at the top, bottom, left, and right
    peripheries of the circles, connected by four 90 degree arcs. It is
    parametrized by the length of the entire network and the number of lanes
    and speed limit of the edges.

    Requires from net_params:

    * **length** : length of the circle
    * **lanes** : number of lanes in the circle
    * **speed_limit** : max speed limit of the circle
    * **resolution** : number of nodes resolution

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import RingNetwork
    >>>
    >>> network = RingNetwork(
    >>>     name='ring_road',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'length': 230,
    >>>             'lanes': 1,
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
        """Initialize a ring network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        r = length / (2 * pi)

        nodes = [{
            "id": "bottom",
            "x": 0,
            "y": -r
        }, {
            "id": "right",
            "x": r,
            "y": 0
        }, {
            "id": "top",
            "x": 0,
            "y": r
        }, {
            "id": "left",
            "x": -r,
            "y": 0
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        r = length / (2 * pi)
        edgelen = length / 4.

        edges = [{
            "id":
                "bottom",
            "type":
                "edgeType",
            "from":
                "bottom",
            "to":
                "right",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(-pi / 2, 0, resolution)
                ]
        }, {
            "id":
                "right",
            "type":
                "edgeType",
            "from":
                "right",
            "to":
                "top",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(0, pi / 2, resolution)
                ]
        }, {
            "id":
                "top",
            "type":
                "edgeType",
            "from":
                "top",
            "to":
                "left",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(pi / 2, pi, resolution)
                ]
        }, {
            "id":
                "left",
            "type":
                "edgeType",
            "from":
                "left",
            "to":
                "bottom",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t), r * sin(t))
                    for t in linspace(pi, 3 * pi / 2, resolution)
                ]
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
            "top": ["top", "left", "bottom", "right"],
            "left": ["left", "bottom", "right", "top"],
            "bottom": ["bottom", "right", "top", "left"],
            "right": ["right", "top", "left", "bottom"]
        }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        ring_length = self.net_params.additional_params["length"]

        edgestarts = [("bottom", 0),
                      ("right", 0.25 * ring_length),
                      ("top", 0.5 * ring_length),
                      ("left", 0.75 * ring_length)]

        return edgestarts
