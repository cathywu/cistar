"""Script containing the base network kernel class."""
import flow.config as config
import json
import subprocess
import os.path as osp
import os
import platform
import time
from flow.core.kernel.network.base import BaseKernelNetwork
from copy import deepcopy

# length of vehicles in the network, in meters
VEHICLE_LENGTH = 5


class AimsunKernelNetwork(BaseKernelNetwork):
    """Network kernel for Aimsun-based simulations.

    This class is responsible for passing features to and calling the
    "generate.py" file within flow/utils/aimsun/. All other features are
    designed to extend BaseKernelNetwork.

    Attributes
    ----------
    kernel_api : any
        an API that may be used to interact with the simulator
    network : flow.networks.Network
        an object containing relevant network-specific features such as the
        locations and properties of nodes and edges in the network
    rts : dict
        specifies routes vehicles can take. See the parent class for
        description of the attribute.
    aimsun_proc : subprocess.Popen
        an object which is used to start or shut down Aimsun from the script
    """

    def __init__(self, master_kernel, sim_params):
        """See parent class."""
        BaseKernelNetwork.__init__(self, master_kernel, sim_params)

        self.kernel_api = None
        self.network = None
        self._edges = None
        self._edge_list = None
        self._junction_list = None
        self.__max_speed = None
        self.__length = None
        self.rts = None
        self._edge_flow2aimsun = {}
        self._edge_aimsun2flow = {}
        self.aimsun_proc = None

    def generate_network(self, network):
        """See parent class."""
        self.network = network

        output = {
            "edges": network.edges,
            "nodes": network.nodes,
            "types": network.types,
            "connections": network.connections,
            "inflows": None,
            "vehicle_types": network.vehicles.types,
            "osm_path": network.net_params.osm_path,
            'render': self.sim_params.render,
            "sim_step": self.sim_params.sim_step,
            "traffic_lights": None,
            "network_name": self.sim_params.network_name,
            "experiment_name": self.sim_params.experiment_name,
            "replication_name": self.sim_params.replication_name,
            "centroid_config_name": self.sim_params.centroid_config_name,
            "subnetwork_name": self.sim_params.subnetwork_name
        }

        if network.net_params.inflows is not None:
            output["inflows"] = network.net_params.inflows.__dict__

        if network.traffic_lights is not None:
            output["traffic_lights"] = network.traffic_lights.__dict__

        cur_dir = os.path.join(config.PROJECT_PATH,
                               'flow/core/kernel/network')
        # TODO: add current time
        with open(os.path.join(cur_dir, 'data_%s.json' % self.sim_params.port), 'w') as outfile:
            json.dump(output, outfile, sort_keys=True, indent=4)

        # path to the Aimsun_Next binary
        if platform.system() == 'Darwin':  # OS X
            binary_name = 'Aimsun Next'
        else:
            binary_name = 'Aimsun_Next'
        aimsun_path = osp.join(osp.expanduser(config.AIMSUN_NEXT_PATH),
                               binary_name)

        # remove network data file if if still exists from
        # the previous simulation
        data_file = 'flow/core/kernel/network/network_data_%s.json' % self.sim_params.port
        data_file_path = os.path.join(config.PROJECT_PATH, data_file)
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
        check_file = 'flow/core/kernel/network/network_data_check%s' % self.sim_params.port
        check_file_path = os.path.join(config.PROJECT_PATH, check_file)
        if os.path.exists(check_file_path):
            os.remove(check_file_path)

        # we need to make flow directories visible to aimsun's python2.7
        os.environ["PYTHONPATH"] = config.PROJECT_PATH
        # path to the supplementary file that is used to generate an aimsun
        # network from a template
        template_path = network.net_params.template
        if template_path is None:
            script_path = osp.join(config.PROJECT_PATH,
                                   'flow/utils/aimsun/generate.py')
        else:
            script_path = osp.join(config.PROJECT_PATH,
                                   'flow/utils/aimsun/load.py')
            file_path = osp.join(config.PROJECT_PATH,
                                 'flow/utils/aimsun/aimsun_template_path_%s' % self.sim_params.port)
            with open(file_path, 'w') as f:
                f.write("%s_%s" % (template_path, self.sim_params.port))
            # instances must have unique template paths to avoid crashing?
            os.popen('cp %s %s_%s' % (template_path, template_path, self.sim_params.port))

        # start the aimsun process
        aimsun_call = [aimsun_path, "-script", script_path, str(self.sim_params.port)]
        self.aimsun_proc = subprocess.Popen(aimsun_call)

        # merge types into edges
        if network.net_params.osm_path is None:
            if network.net_params.template is None:
                for i in range(len(network.edges)):
                    if 'type' in network.edges[i]:
                        for typ in network.types:
                            if typ['id'] == network.edges[i]['type']:
                                new_dict = deepcopy(typ)
                                new_dict.pop("id")
                                network.edges[i].update(new_dict)
                                break

                self._edges = {}
                for edge in deepcopy(network.edges):
                    edge_name = edge['id']
                    self._edges[edge_name] = {}
                    del edge['id']
                    self._edges[edge_name] = edge

                # list of edges and internal links (junctions)
                self._edge_list = [
                    edge_id for edge_id in self._edges.keys()
                    if edge_id[0] != ':'
                ]
                self._junction_list = list(
                    set(self._edges.keys()) - set(self._edge_list))

            else:
                # load network from template
                scenar_file = "flow/core/kernel/network/network_data_%s.json" % self.sim_params.port
                scenar_path = os.path.join(config.PROJECT_PATH, scenar_file)

                check_file = "flow/core/kernel/network/network_data_check_%s" % self.sim_params.port
                check_path = os.path.join(config.PROJECT_PATH, check_file)

                # a check file is created when all the network data
                # have been written ; it is necessary since writing
                # all the data can take several seconds for large networks
                while not os.path.exists(check_path):
                    time.sleep(0.1)
                os.remove(check_path)

                # network_data.json has been written, load its content
                with open(scenar_path) as f:
                    content = json.load(f)
                os.remove(scenar_path)

                self._edges = content['sections']
                self._edge_list = self._edges.keys()
                self._junction_list = content['turnings']
                # TODO load everything that is in content into the network

        else:
            data_file = 'flow/utils/aimsun/osm_edges_%s.json' % self.sim_params.port
            filepath = os.path.join(config.PROJECT_PATH, data_file)

            while not os.path.exists(filepath):
                time.sleep(0.5)

            with open(filepath) as f:
                self._edges = json.load(f)
            # list of edges and internal links (junctions)
            self._edge_list = [
                edge_id for edge_id in self._edges.keys()
                if edge_id[0] != ':'
            ]
            self._junction_list = list(
                set(self._edges.keys()) - set(self._edge_list))

            # delete the file
            os.remove(filepath)

        # maximum achievable speed on any edge in the network
        self.__max_speed = max(
            self.speed_limit(edge) for edge in self.get_edge_list())

        # length of the network, or the portion of the network in
        # which cars are meant to be distributed
        self.__length = sum(
            self.edge_length(edge_id) for edge_id in self.get_edge_list()
        )

        # parameters to be specified under each unique subclass's
        # __init__() function
        self.edgestarts = self.network.edge_starts

        # if no edge_starts are specified, generate default values to be used
        # by the "get_edge" method
        if self.edgestarts is None:
            length = 0
            self.edgestarts = []
            for edge_id in sorted(self._edge_list):
                # the current edge starts where the last edge ended
                self.edgestarts.append((edge_id, length))
                # increment the total length of the network with the length of
                # the current edge
                length += self._edges[edge_id]['length']
        self.edgestarts.sort(key=lambda tup: tup[1])

        # these optional parameters need only be used if "no-internal-links"
        # is set to "false" while calling sumo's netconvert function
        self.internal_edgestarts = self.network.internal_edge_starts
        self.internal_edgestarts_dict = dict(self.internal_edgestarts)

        # total_edgestarts and total_edgestarts_dict contain all of the above
        # edges, with the former being ordered by position
        self.total_edgestarts = self.edgestarts + self.internal_edgestarts
        self.total_edgestarts.sort(key=lambda tup: tup[1])

        self.total_edgestarts_dict = dict(self.total_edgestarts)

        # specify routes vehicles can take  # TODO: move into a method
        self.rts = self.network.routes

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

        # create the edge mapping from edges names in Flow to Aimsun and vice
        # versa
        self._edge_flow2aimsun = {}
        self._edge_aimsun2flow = {}

        for edge in self.get_edge_list():
            aimsun_edge = self.kernel_api.get_edge_name(edge)
            self._edge_flow2aimsun[edge] = aimsun_edge
            self._edge_aimsun2flow[aimsun_edge] = edge

    def update(self, reset):
        """See parent class."""
        pass

    def close(self):
        """See parent class."""
        # delete the json file that was used to read the network data
        cur_dir = os.path.join(config.PROJECT_PATH,
                               'flow/core/kernel/network')
        os.remove(os.path.join(cur_dir, 'data_%s.json' % self.sim_params.port))
        if self.network.net_params.template is not None:
            os.remove('%s_%s' % (self.network.net_params.template,
                                 self.sim_params.port))

    ###########################################################################
    #                        State acquisition methods                        #
    ###########################################################################

    def edge_length(self, edge_id):
        """See parent class."""
        try:
            return self._edges[edge_id]["length"]
        except KeyError:
            print('Error in edge length with key', edge_id)
            return -1001

    def length(self):
        """See parent class."""
        return sum(self.edge_length(edge_id)
                   for edge_id in self.get_edge_list())

    def non_internal_length(self):
        """See parent class."""
        return sum(self.edge_length(edge_id)
                   for edge_id in self.get_edge_list())

    def speed_limit(self, edge_id):
        """See parent class."""
        try:
            return self._edges[edge_id]["speed"]
        except KeyError:
            print('Error in speed limit with key', edge_id)
            return -1001

    def max_speed(self):
        """See parent class."""
        return max(
            self.speed_limit(edge) for edge in self.get_edge_list())

    def num_lanes(self, edge_id):
        """See parent class."""
        try:
            return self._edges[edge_id]["numLanes"]
        except KeyError:
            print('Error in num lanes with key', edge_id)
            return -1001

    def get_edge_list(self):
        """See parent class."""
        return self._edge_list

    def get_junction_list(self):
        """See parent class."""
        return self._junction_list

    def _get_edge(self, x):  # TODO: maybe remove
        """See parent class."""
        for (edge, start_pos) in reversed(self.edgestarts):
            if x >= start_pos:
                return edge, x - start_pos

    def get_x(self, edge, position):  # TODO: maybe remove
        """See parent class."""
        # if there was a collision which caused the vehicle to disappear,
        # return an x value of -1001
        if len(edge) == 0:
            return -1001

        if edge[0] == ":" or '_to_' in edge:
            try:
                return self.internal_edgestarts_dict[edge] + position
            except KeyError:
                # in case several internal links are being generalized for
                # by a single element (for backwards compatibility)
                edge_name = edge.rsplit("_", 1)[0]
                return self.total_edgestarts_dict.get(edge_name, -1001)
        else:
            return self.total_edgestarts_dict[edge] + position

    def next_edge(self, edge, lane):
        """See parent class."""
        try:
            return self._connections["next"][edge][lane]
        except KeyError:
            return []

    def prev_edge(self, edge, lane):
        """See parent class."""
        try:
            return self._connections["prev"][edge][lane]
        except KeyError:
            return []

    def aimsun_edge_name(self, edge):
        """Return the edge name in Aimsun."""
        return self._edge_flow2aimsun[edge]

    def flow_edge_name(self, edge):
        """Return the edge name in Aimsun."""
        if edge not in self._edge_aimsun2flow:
            # print("aimsun edge unknown: {}".format(edge))
            return ''
        else:
            return self._edge_aimsun2flow[edge]
