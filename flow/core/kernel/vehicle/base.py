"""Script containing the base vehicle kernel class."""

from abc import ABCMeta, abstractmethod


class KernelVehicle(object, metaclass=ABCMeta):
    """Flow vehicle kernel.

    This kernel sub-class is used to interact with the simulator with regards
    to all vehicle-dependent components. Specifically, this class contains
    methods for:

    * Interacting with the simulator: This includes apply acceleration, lane
      change, and routing commands to specific vehicles in the simulator. In
      addition, methods exist to add or remove a specific vehicle from the
      network, and update internal state information after every simulation
      step in order to support and potentially speed up all state-acquisition
      methods.
    * Visually distinguishing vehicles by type: In the case when some vehicles
      are controlled by a reinforcement learning agent or some other
      controller, these methods can be used to visually distinguish the
      vehicles during rendering by RL/actuated, human-observed, and
      human-unobserved. The traci simulator, for instance, renders RL vehicles
      as red, observed human vehicles as cyan, and unobserved human vehicles as
      white. In the absence of RL/actuated agents, all vehicles are white.
    * State acquisition: Finally, this methods contains several methods for
      acquiring state information from specific vehicles. For example, if you
      would like to get the speed of a vehicle from the environment, that can
      be done by calling:

        >>> from flow.envs.base import Env
        >>> env = Env(...)
        >>> veh_id = "test_car"  # name of the vehicle
        >>> speed = env.k.vehicle.get_speed(veh_id)

    All methods in this class are abstract, and must be filled in by the child
    vehicle kernel of separate simulators.
    """

    def __init__(self,
                 master_kernel,
                 sim_params):
        """Instantiate the Flow vehicle kernel.

        Parameters
        ----------
        master_kernel : flow.core.kernel.Kernel
            the higher level kernel (used to call methods from other
            sub-kernels)
        sim_params : flow.core.params.SimParams
            simulation-specific parameters
        """
        self.master_kernel = master_kernel
        self.kernel_api = None
        self.sim_step = sim_params.sim_step

    def pass_api(self, kernel_api):
        """Acquire the kernel api that was generated by the simulation kernel.

        Parameters
        ----------
        kernel_api : any
            an API that may be used to interact with the simulator
        """
        self.kernel_api = kernel_api

    ###########################################################################
    #               Methods for interacting with the simulator                #
    ###########################################################################

    @abstractmethod
    def update(self, reset):
        """Update the vehicle kernel with data from the current time step.

        This method is used to optimize the computational efficiency of
        acquiring vehicle state information from the kernel.

        Parameters
        ----------
        reset : bool
            specifies whether the simulator was reset in the last simulation
            step
        """
        pass

    @abstractmethod
    def add(self, veh_id, type_id, edge, pos, lane, speed):
        """Add a vehicle to the network.

        Parameters
        ----------
        veh_id : str
            unique identifier of the vehicle to be added
        type_id : str
            vehicle type of the added vehicle
        edge : str
            starting edge of the added vehicle
        pos : float
            starting position of the added vehicle
        lane : int
            starting lane of the added vehicle
        speed : float
            starting speed of the added vehicle
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset any additional state that needs to be reset."""
        pass

    @abstractmethod
    def remove(self, veh_id):
        """Remove a vehicle.

        This method removes all traces of the vehicle from the vehicles kernel
        and all valid ID lists, and decrements the total number of vehicles in
        this class.

        In addition, if the vehicle is still in the network, this method calls
        the necessary simulator-specific commands to remove it.

        Parameters
        ----------
        veh_id : str
            unique identifier of the vehicle to be removed
        """
        pass

    @abstractmethod
    def apply_acceleration(self, veh_id, acc):
        """Apply the acceleration requested by a vehicle in the simulator.

        Parameters
        ----------
        veh_id : str or list of str
            list of vehicle identifiers
        acc : float or array_like
            requested accelerations from the vehicles
        """
        pass

    @abstractmethod
    def apply_lane_change(self, veh_id, direction):
        """Apply an instantaneous lane-change to a set of vehicles.

        This method also prevents vehicles from moving to lanes that do not
        exist, and set the "last_lc" variable for RL vehicles that lane changed
        to match the current time step, in order to assist in maintaining a
        lane change duration for these vehicles.

        Parameters
        ----------
        veh_id : str or list of str
            list of vehicle identifiers
        direction : {-1, 0, 1} or list of {-1, 0, 1}
            -1: lane change to the right
             0: no lane change
             1: lane change to the left

        Raises
        ------
        ValueError
            If any of the direction values are not -1, 0, or 1.
        """
        pass

    @abstractmethod
    def choose_routes(self, veh_id, route_choices):
        """Update the route choice of vehicles in the network.

        Parameters
        ----------
        veh_id : str or list of str
            list of vehicle identifiers
        route_choices : array_like
            list of edges the vehicle wishes to traverse, starting with the
            edge the vehicle is currently on. If a value of None is provided,
            the vehicle does not update its route
        """
        pass

    @abstractmethod
    def set_max_speed(self, veh_id, max_speed):
        """Update the maximum allowable speed by a vehicles in the network.

        Parameters
        ----------
        veh_id : list
            vehicle identifier
        max_speed : float
            desired max speed by the vehicle
        """
        pass

    ###########################################################################
    # Methods to visually distinguish vehicles by {RL, observed, unobserved}  #
    ###########################################################################

    @abstractmethod
    def update_vehicle_colors(self):
        """Modify the color of vehicles if rendering is active."""
        pass

    @abstractmethod
    def set_observed(self, veh_id):
        """Add a vehicle to the list of observed vehicles."""
        pass

    @abstractmethod
    def remove_observed(self, veh_id):
        """Remove a vehicle from the list of observed vehicles."""
        pass

    @abstractmethod
    def get_observed_ids(self):
        """Return the list of observed vehicles."""
        pass

    @abstractmethod
    def get_color(self, veh_id):
        """Return and RGB tuple of the color of the specified vehicle."""
        pass

    @abstractmethod
    def set_color(self, veh_id, color):
        """Set the color of the specified vehicle with the RGB tuple."""
        pass

    ###########################################################################
    #                        State acquisition methods                        #
    ###########################################################################

    @abstractmethod
    def get_orientation(self, veh_id):
        """Return the orientation of the vehicle of veh_id."""
        pass

    @abstractmethod
    def get_timestep(self, veh_id):
        """Return the time step of the vehicle of veh_id."""
        pass

    @abstractmethod
    def get_timedelta(self, veh_id):
        """Return the simulation time delta of the vehicle of veh_id."""
        pass

    @abstractmethod
    def get_type(self, veh_id):
        """Return the type of the vehicle of veh_id."""
        pass

    @abstractmethod
    def get_ids(self):
        """Return the names of all vehicles currently in the network."""
        pass

    @abstractmethod
    def get_human_ids(self):
        """Return the names of all non-rl vehicles currently in the network."""
        pass

    @abstractmethod
    def get_controlled_ids(self):
        """Return the names of all flow acceleration-controlled vehicles.

        This only include vehicles that are currently in the network.
        """
        pass

    @abstractmethod
    def get_controlled_lc_ids(self):
        """Return the names of all flow lane change-controlled vehicles.

        This only include vehicles that are currently in the network.
        """
        pass

    @abstractmethod
    def get_rl_ids(self):
        """Return the names of all rl-controlled vehicles in the network."""
        pass

    @abstractmethod
    def get_ids_by_edge(self, edges):
        """Return the names of all vehicles in the specified edge.

        If no vehicles are currently in the edge, then returns an empty list.
        """
        pass

    @abstractmethod
    def get_inflow_rate(self, time_span):
        """Return the inflow rate (in veh/hr) of vehicles from the network.

        This value is computed over the specified **time_span** seconds.
        """
        pass

    @abstractmethod
    def get_outflow_rate(self, time_span):
        """Return the outflow rate (in veh/hr) of vehicles from the network.

        This value is computed over the specified **time_span** seconds.
        """
        pass

    @abstractmethod
    def get_num_arrived(self):
        """Return the number of vehicles that arrived in the last time step."""
        pass

    @abstractmethod
    def get_arrived_ids(self):
        """Return the ids of vehicles that arrived in the last time step."""
        pass

    @abstractmethod
    def get_departed_ids(self):
        """Return the ids of vehicles that departed in the last time step."""
        pass

    @abstractmethod
    def get_num_not_departed(self):
        """Return the number of vehicles not departed in the last time step.

        This includes vehicles that were loaded but not departed.
        """
        pass

    @abstractmethod
    def get_fuel_consumption(self, veh_id, error=-1001):
        """Return the mpg / s of the specified vehicle.
        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found
        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def get_speed(self, veh_id, error=-1001):
        """Return the speed of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def get_default_speed(self, veh_id, error=-1001):
        """Return the expected speed if no control were applied.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def get_position(self, veh_id, error=-1001):
        """Return the position of the vehicle relative to its current edge.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def get_edge(self, veh_id, error=""):
        """Return the edge the specified vehicle is currently on.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        str
        """
        pass

    @abstractmethod
    def get_lane(self, veh_id, error=-1001):
        """Return the lane index of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def get_route(self, veh_id, error=list()):
        """Return the route of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list of str
        """
        pass

    @abstractmethod
    def get_length(self, veh_id, error=-1001):
        """Return the length of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def get_leader(self, veh_id, error=""):
        """Return the leader of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        str
        """
        pass

    @abstractmethod
    def get_follower(self, veh_id, error=""):
        """Return the follower of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        str
        """
        pass

    @abstractmethod
    def get_headway(self, veh_id, error=-1001):
        """Return the headway of the specified vehicle(s).

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def get_last_lc(self, veh_id, error=-1001):
        """Return the last time step a vehicle changed lanes.

        Note: This value is only stored for RL vehicles. All other vehicles
        calling this will cause a warning to be printed and their "last_lc"
        term will be the error value.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def get_acc_controller(self, veh_id, error=None):
        """Return the acceleration controller of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        object
        """
        pass

    @abstractmethod
    def get_lane_changing_controller(self, veh_id, error=None):
        """Return the lane changing controller of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        object
        """
        pass

    @abstractmethod
    def get_routing_controller(self, veh_id, error=None):
        """Return the routing controller of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        object
        """
        pass

    @abstractmethod
    def get_lane_headways(self, veh_id, error=list()):
        """Return the lane headways of the specified vehicles.

        This includes the headways between the specified vehicle and the
        vehicle immediately ahead of it in all lanes.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list of float
        """
        pass

    @abstractmethod
    def get_lane_leaders_speed(self, veh_id, error=list()):
        """Return the speed of the leaders of the specified vehicles.

        This includes the speed between the specified vehicle and the
        vehicle immediately ahead of it in all lanes.

        Missing lead vehicles have a speed of zero.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list of float
        """
        pass

    @abstractmethod
    def get_lane_followers_speed(self, veh_id, error=list()):
        """Return the speed of the followers of the specified vehicles.

        This includes the speed between the specified vehicle and the
        vehicle immediately behind it in all lanes.

        Missing following vehicles have a speed of zero.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list of float
        """
        pass

    @abstractmethod
    def get_lane_leaders(self, veh_id, error=list()):
        """Return the leaders for the specified vehicle in all lanes.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        lis of str
        """
        pass

    @abstractmethod
    def get_lane_tailways(self, veh_id, error=list()):
        """Return the lane tailways of the specified vehicle.

        This includes the headways between the specified vehicle and the
        vehicle immediately behind it in all lanes.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list of float
        """
        pass

    @abstractmethod
    def get_lane_followers(self, veh_id, error=list()):
        """Return the followers for the specified vehicle in all lanes.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : list, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list of str
        """
        pass

    @abstractmethod
    def get_x_by_id(self, veh_id):
        """Provide a 1-D representation of the position of a vehicle.

        Note: These values are only meaningful if the specify_edge_starts
        method in the network is set appropriately; otherwise, a value of 0 is
        returned for all vehicles.

        Parameters
        ----------
        veh_id : str
            vehicle identifier

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def get_max_speed(self, veh_id, error):
        """Return the max speed of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float
        """
        pass
