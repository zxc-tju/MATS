class PredictionSettings:
    def __init__(self, mats, hyperparams, env, num_modes):
        self.mats = mats
        self.hyperparams = hyperparams
        self.env = env
        self.num_modes = num_modes


class Node:
    def __init__(self, x, y, theta, v, first_timestep, last_timestep, type_, id_):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.first_timestep = first_timestep
        self.last_timestep = last_timestep
        self.type = type_
        self.id = id_


class Scene:
    def __init__(self, robot, non_robot_nodes, dt, timesteps, node_ids, x_offset, y_offset, scene_name):
        self.robot = robot
        self.non_robot_nodes = non_robot_nodes
        self.dt = dt
        self.timesteps = timesteps
        self.node_ids = node_ids
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.name = scene_name


def create_scene(env, scene_num: int):
    # information about agents in scene
    robot = None
    non_robot_nodes = []
    x_offset = env.scenes[scene_num].x_min
    y_offset = env.scenes[scene_num].y_min
    scene_name = env.scenes[scene_num].name
    for node in env.scenes[scene_num].nodes:
        if node.is_robot:
            x = [item[0] + x_offset for item in node.data.data]
            y = [item[1] + y_offset for item in node.data.data]
            theta = [item[8] for item in node.data.data]
            v = [item[10] for item in node.data.data]
            first_timestep = node.first_timestep
            last_timestep = node.last_timestep
            type_ = node.type.name
            id_ = node.id
            robot = Node(x, y, theta, v, first_timestep, last_timestep, type_, id_)
        else:
            x = [item[0] + x_offset for item in node.data.data]
            y = [item[1] + y_offset for item in node.data.data]
            theta = []
            v = []
            first_timestep = node.first_timestep
            last_timestep = node.last_timestep
            type_ = node.type.name
            id_ = node.id
            non_robot_nodes.append(Node(x, y, theta, v, first_timestep, last_timestep, type_, id_))

    # possibly useful metadata
    dt = env.scenes[scene_num].dt
    timesteps = env.scenes[scene_num].timesteps
    node_ids = [str(node.id) for node in non_robot_nodes]

    return Scene(robot, non_robot_nodes, dt, timesteps, node_ids, x_offset, y_offset, scene_name)


def get_scene_info(env, scene_num: int):
    # information about agents in scene
    robot = None
    non_robot_nodes = []
    x_offset = env.scenes[scene_num].x_min
    y_offset = env.scenes[scene_num].y_min
    scene_name = env.scenes[scene_num].name
    for node in env.scenes[scene_num].nodes:
        if node.is_robot:
            x = [item[0] + x_offset for item in node.data.data]
            y = [item[1] + y_offset for item in node.data.data]
            theta = [item[8] for item in node.data.data]
            v = [item[10] for item in node.data.data]
            first_timestep = node.first_timestep
            last_timestep = node.last_timestep
            type_ = node.type.name
            id_ = node.id
            robot = Node(x, y, theta, v, first_timestep, last_timestep, type_, id_)
        else:
            x = [item[0] + x_offset for item in node.data.data]
            y = [item[1] + y_offset for item in node.data.data]
            theta = []
            v = []
            first_timestep = node.first_timestep
            last_timestep = node.last_timestep
            type_ = node.type.name
            id_ = node.id
            non_robot_nodes.append(Node(x, y, theta, v, first_timestep, last_timestep, type_, id_))

    # possibly useful metadata
    dt = env.scenes[scene_num].dt
    timesteps = env.scenes[scene_num].timesteps
    node_ids = [str(node.id) for node in non_robot_nodes]

    return robot, non_robot_nodes, node_ids


class ControlLimits:
    def __init__(self, omega_max, omega_min, a_max, a_min, vs_max, vs_min):
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.a_max = a_max
        self.a_min = a_min
        self.vs_max = vs_max
        self.vs_min = vs_min


class DynamicsModel:
    def __init__(self, control_dim, state_dim, control_limits_obj):
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.u_limit = control_limits_obj


class Obstacle:
    def __init__(self, positions, active=False):
        self.positions = positions
        self.active = active


def init_node_obstacles(node_ids, vals):
    for node_id in node_ids:
        obstacle_positions = [[[0., 0.] for _ in range(vals.obstacle_horizon)] for _ in range(vals.num_modes)]
        vals.obstacles[node_id] = Obstacle(obstacle_positions)
