from stljax.formula import *
import jax.numpy as jnp
from functools import partial

# NOTE
# If using Expressions to define formulas, `stljax` expects input signals to be of size `[time_dim]`.
# If using Predicates to define formulas, `stljax` expects input signals to be of size `[time_dim, state_dim]` where `state_dim` is the expected input size of your predicate function.


def get(traj, ind=0):
    return traj[:, ind]


# for team vs team games, trajectories are of:
# (bs, time_dim, state_dim * num_agents)
# agents are concatenated as (ego_0_x, ego_0_y, ...,  ego_1_x, ego_1_y, ..., ..., opp_0_x, ..., opp_1_x, ..., ...)
def get_joint(traj, agent_ind=0, state_ind=0, state_dim=6):
    flattened_ind = agent_ind * state_dim + state_ind
    return traj[:, flattened_ind]


def radius(traj, x_ind, y_ind, c):
    return jnp.linalg.norm(traj[:, [x_ind, y_ind]] - jnp.array(c), axis=-1)


def radius_collision(traj, x_ind, y_ind, cx_ind, cy_ind):
    return jnp.linalg.norm(traj[:, [x_ind, y_ind]] - traj[:, [cx_ind, cy_ind]], axis=-1)


def radius3d(x, y, z, c):
    return jnp.linalg.norm(jnp.array([x, y, z]) - jnp.array(c), axis=-1)


def radius3d_collision(traj, x_ind, y_ind, z_ind, cx_ind, cy_ind, cz_ind):
    return jnp.linalg.norm(
        traj[:, [x_ind, y_ind, z_ind]] - traj[:, [cx_ind, cy_ind, cz_ind]], axis=-1
    )


# def radius3d_collision_joint(
#     traj, ego_ind, opp_ind, x_ind, y_ind, z_ind, cx_ind, cy_ind, cz_ind
# ):
#     return jnp.linalg.norm(
#         traj[ego_ind, :, [x_ind, y_ind, z_ind]]
#         - traj[opp_ind, :, [cx_ind, cy_ind, cz_ind]],
#         axis=-1,
#     )


def get_inside_box_predicate(obs):
    # obstacle is of shape (4, )
    # (x1, x2, y1, y2)
    x = Predicate("x", predicate_function=partial(get, ind=0))
    y = Predicate("y", predicate_function=partial(get, ind=1))
    return (x > obs[0]) & (x < obs[1]) & (y > obs[2]) & (y < obs[3])


def get_inside_box_3d_predicate(obs):
    # obstacle is of shape (6, )
    # (x1, x2, y1, y2, z1, z2)
    x = Predicate("x", predicate_function=partial(get, ind=0))
    y = Predicate("y", predicate_function=partial(get, ind=1))
    z = Predicate("z", predicate_function=partial(get, ind=2))
    return (((x > obs[0]) & (x < obs[1])) & ((y > obs[2]) & (y < obs[3]))) & (
        (z > obs[4]) & (z < obs[5])
    )


def get_altitude_rule_3d_predicate(zone_bounds, altitude_bounds, x_ind=0, z_ind=2):
    x = Predicate("x", predicate_function=partial(get, ind=x_ind))
    z = Predicate("z", predicate_function=partial(get, ind=z_ind))
    zone_lower_bound = zone_bounds[0]
    zone_upper_bound = zone_bounds[1]
    altitude_lower_bound = altitude_bounds[0]
    altitude_upper_bound = altitude_bounds[1]
    formula = Implies(
        (x >= zone_lower_bound) & (x <= zone_upper_bound),
        (z >= altitude_lower_bound) & (z <= altitude_upper_bound),
    )
    return formula


def get_altitude_rule_3d_predicate_joint(
    zone_bounds, altitude_bounds, agent_ind, x_ind=0, z_ind=2
):
    x = Predicate(
        "x", predicate_function=partial(get_joint, agent_ind=agent_ind, state_ind=x_ind)
    )
    z = Predicate(
        "z", predicate_function=partial(get_joint, agent_ind=agent_ind, state_ind=z_ind)
    )
    zone_lower_bound = zone_bounds[0]
    zone_upper_bound = zone_bounds[1]
    altitude_lower_bound = altitude_bounds[0]
    altitude_upper_bound = altitude_bounds[1]
    formula = Implies(
        (x >= zone_lower_bound) & (x <= zone_upper_bound),
        (z >= altitude_lower_bound) & (z <= altitude_upper_bound),
    )
    return formula


def get_altitude_rule_3d_predicate_cfmjx(min_z, max_z, agent_ind, z_ind=2):
    z = Predicate(
        "z", predicate_function=partial(get_joint, agent_ind=agent_ind, state_ind=z_ind)
    )
    formula = Implies((z >= min_z), (z <= max_z))
    return formula


def get_always_outside_circle_predicate(obs):
    # obs is of shape (3, )
    # (x, y, r)
    # x = Predicate("x", predicate_function=partial(get, ind=0))
    # y = Predicate("y", predicate_function=partial(get, ind=1))
    r = Predicate("r", predicate_function=partial(radius, x_ind=0, y_ind=1, c=obs[:2]))
    return Always(r > obs[2])


def get_always_outside_box_3d_predicate(obs):
    # obs is of shape (6, )
    # (x1, x2, y1, y2, z1, z2)
    x = Predicate("x", predicate_function=partial(get, ind=0))
    y = Predicate("y", predicate_function=partial(get, ind=1))
    z = Predicate("z", predicate_function=partial(get, ind=2))
    return Always(
        (((x < obs[0]) | (x > obs[1])) | ((y < obs[2]) | (y > obs[3])))
        | ((z < obs[4]) | (z > obs[5]))
    )


def get_always_outside_box_3d_predicate_joint(obs, agent_ind=0):
    # obs is of shape (6, )
    # (x1, x2, y1, y2, z1, z2)
    x = Predicate(
        "x", predicate_function=partial(get_joint, agent_ind=agent_ind, state_ind=0)
    )
    y = Predicate(
        "y", predicate_function=partial(get_joint, agent_ind=agent_ind, state_ind=1)
    )
    z = Predicate(
        "z", predicate_function=partial(get_joint, agent_ind=agent_ind, state_ind=2)
    )
    return Always(
        (((x < obs[0]) | (x > obs[1])) | ((y < obs[2]) | (y > obs[3])))
        | ((z < obs[4]) | (z > obs[5]))
    )


def get_inside_box_3d_predicate_joint(obs, agent_ind=0):
    # obs is of shape (n, 6)
    # (x1, x2, y1, y2, z1, z2)
    # traj expected of shape (n, t, 6)
    x = Predicate(
        "x", predicate_function=partial(get_joint, agent_ind=agent_ind, state_ind=0)
    )
    y = Predicate(
        "y", predicate_function=partial(get_joint, agent_ind=agent_ind, state_ind=1)
    )
    z = Predicate(
        "z", predicate_function=partial(get_joint, agent_ind=agent_ind, state_ind=2)
    )
    return (((x > obs[0]) & (x < obs[1])) & ((y > obs[2]) & (y < obs[3]))) & (
        (z > obs[4]) & (z < obs[5])
    )


def get_always_no_collision_between_agents_3d_predicate(
    agent2_x_idx, safe_distance=0.09
):
    d = Predicate(
        "d",
        predicate_function=partial(
            radius3d_collision,
            x_ind=0,
            y_ind=1,
            z_ind=2,
            cx_ind=agent2_x_idx,
            cy_ind=agent2_x_idx + 1,
            cz_ind=agent2_x_idx + 2,
        ),
    )
    return Always(d > safe_distance)


def get_always_no_collision_between_agents_3d_predicate_joint(
    ego_ind, opp_ind, state_dim=6, safe_distance=0.09
):
    d = Predicate(
        "d",
        predicate_function=partial(
            radius3d_collision,
            x_ind=ego_ind * state_dim + 0,
            y_ind=ego_ind * state_dim + 1,
            z_ind=ego_ind * state_dim + 2,
            cx_ind=opp_ind * state_dim + 0,
            cy_ind=opp_ind * state_dim + 1,
            cz_ind=opp_ind * state_dim + 2,
        ),
    )
    return Always(d > safe_distance)


def get_collision_between_agents_3d_predicate(agent2_x_idx, safe_distance=0.09):
    d = Predicate(
        "d",
        predicate_function=partial(
            radius3d_collision,
            x_ind=0,
            y_ind=1,
            z_ind=2,
            cx_ind=agent2_x_idx,
            cy_ind=agent2_x_idx + 1,
            cz_ind=agent2_x_idx + 2,
        ),
    )
    return Eventually(d <= safe_distance)


def get_collision_between_agents_3d_predicate_joint(
    ego_ind, opp_ind, state_dim=6, safe_distance=0.09
):
    d = Predicate(
        "d",
        predicate_function=partial(
            radius3d_collision,
            x_ind=ego_ind * state_dim + 0,
            y_ind=ego_ind * state_dim + 1,
            z_ind=ego_ind * state_dim + 2,
            cx_ind=opp_ind * state_dim + 0,
            cy_ind=opp_ind * state_dim + 1,
            cz_ind=opp_ind * state_dim + 2,
        ),
    )
    return Eventually(d <= safe_distance)


def get_always_no_collision_between_agents_predicate(agent2_x_idx, safe_distance=0.09):
    d = Predicate(
        "d",
        predicate_function=partial(
            radius_collision,
            x_ind=0,
            y_ind=1,
            cx_ind=agent2_x_idx,
            cy_ind=agent2_x_idx + 1,
        ),
    )
    return Always(d > safe_distance)


def get_collision_between_agents_predicate(agent2_x_idx, safe_distance=0.09):
    d = Predicate(
        "d",
        predicate_function=partial(
            radius_collision,
            x_ind=0,
            y_ind=1,
            cx_ind=agent2_x_idx,
            cy_ind=agent2_x_idx + 1,
        ),
    )
    return Eventually(d <= safe_distance)


def get_reach_avoid_formula(obs_1, obs_2, obs_3, obs_4, goal, T):
    inside_box_1 = get_inside_box_predicate(obs_1)
    inside_box_2 = get_inside_box_predicate(obs_2)
    reach_goal = get_inside_box_predicate(goal)
    always_outside_circle = get_always_outside_circle_predicate(obs_3)
    has_been_inside_box_1 = Eventually(
        subformula=Always(subformula=inside_box_1, interval=[0, T])
    )
    has_been_inside_box_2 = Eventually(
        subformula=Always(subformula=inside_box_2, interval=[0, T])
    )
    eventually_reach_goal = Eventually(
        subformula=Always(subformula=reach_goal, interval=[0, 1])
    )
    formula = (has_been_inside_box_1 & has_been_inside_box_2) & always_outside_circle
    return formula


def get_reach_avoid_two_agent_formula(
    obs_1, obs_2, obs_3, obs_4, goal, T, agent2_x_idx=3
):
    # ego formula
    inside_box_1 = get_inside_box_predicate(obs_1)
    inside_box_2 = get_inside_box_predicate(obs_2)
    reach_goal = get_inside_box_predicate(goal)
    always_outside_circle = get_always_outside_circle_predicate(obs_3)
    has_been_inside_box_1 = Eventually(
        subformula=Always(subformula=inside_box_1, interval=[0, T])
    )
    has_been_inside_box_2 = Eventually(
        subformula=Always(subformula=inside_box_2, interval=[0, T])
    )
    eventually_reach_goal = Eventually(
        subformula=Always(subformula=reach_goal, interval=[0, 1])
    )
    always_nocol = get_always_no_collision_between_agents_predicate(
        agent2_x_idx, safe_distance=0.09
    )
    ego_formula = (
        (has_been_inside_box_1 & has_been_inside_box_2)
        & always_outside_circle
        & always_nocol
    )

    # opp formula
    col = get_collision_between_agents_predicate(agent2_x_idx, safe_distance=0.09)
    return ego_formula, col


def get_reach_avoid_two_agent_formula_3d(
    obs, goal, T, zone1, zone2, altitude1, altitude2, agent2_x_idx=3
):
    # ego stl formula
    inside_goal_box = get_inside_box_3d_predicate(goal)
    has_been_inside_goal_box = Eventually(
        subformula=Always(subformula=inside_goal_box, interval=[0, T])
    )
    always_stay_outside_unsafe_box_formula = get_always_outside_box_3d_predicate(obs)
    always_no_collision_between_agents_formula = (
        get_always_no_collision_between_agents_3d_predicate(
            agent2_x_idx, safe_distance=0.09
        )
    )
    altitude_rule1 = get_altitude_rule_3d_predicate(zone1, altitude1)
    altitude_rule2 = get_altitude_rule_3d_predicate(zone2, altitude2)
    always_altitude_rule1 = Always(subformula=altitude_rule1, interval=[0, T])
    always_altitude_rule2 = Always(subformula=altitude_rule2, interval=[0, T])
    ego_formula = (
        (has_been_inside_goal_box & always_stay_outside_unsafe_box_formula)
        & (always_altitude_rule1 & always_altitude_rule2)
    ) & always_no_collision_between_agents_formula

    # opp stl formula
    collision_formula = get_collision_between_agents_3d_predicate(
        agent2_x_idx, safe_distance=0.09
    )
    opp_altitude_rule1 = get_altitude_rule_3d_predicate(
        zone1, altitude1, x_ind=agent2_x_idx, z_ind=agent2_x_idx + 2
    )
    opp_altitude_rule2 = get_altitude_rule_3d_predicate(
        zone2, altitude2, x_ind=agent2_x_idx, z_ind=agent2_x_idx + 2
    )
    always_opp_altitude_rule1 = Always(subformula=opp_altitude_rule1, interval=[0, T])
    always_opp_altitude_rule2 = Always(subformula=opp_altitude_rule2, interval=[0, T])
    opp_formula = (
        always_opp_altitude_rule1 & always_opp_altitude_rule2 & collision_formula
    )
    return ego_formula, opp_formula


def get_reach_avoid_two_team_formula_3d(
    obs, goal, T, zone1, zone2, altitude1, altitude2, num_ego, num_opp
):
    # TODO: double check ego and opp indices
    # should be (num_ego + num_opp, time_dim, state_dim)
    # --------------EGO--------------------
    # for ego, disjunction of all ego agents inside each box, then conjunction of all boxes
    goal_reach_list = []
    for goal_i in range(goal.shape[0]):
        inside_list = []
        for ego_i in range(num_ego):
            inside_list.append(
                get_inside_box_3d_predicate_joint(goal[goal_i], agent_ind=ego_i)
            )

        inside_goal = inside_list[0]
        for i in range(1, len(inside_list)):
            inside_goal = Or(inside_goal, inside_list[i])

        goal_reach_list.append(Eventually(inside_goal, interval=[int(T / 2), T]))
        # goal_reach_list.append(Eventually(inside_goal))
    all_goal_reach = goal_reach_list[0]
    for j in range(1, len(goal_reach_list)):
        all_goal_reach = And(all_goal_reach, goal_reach_list[j])

    # all ego must always avoid obs
    avoid_obs_list = []
    for ego_i in range(num_ego):
        avoid_obs_list.append(
            get_always_outside_box_3d_predicate_joint(obs, agent_ind=ego_i)
        )
    all_avoid_obs = avoid_obs_list[0]
    for j in range(1, len(avoid_obs_list)):
        all_avoid_obs = And(all_avoid_obs, avoid_obs_list[j])

    # all ego must always avoid collision with each other
    ego_avoid_ego_list = []
    for ego_i in range(num_ego):
        for ego_j in range(ego_i + 1, num_ego):
            ego_avoid_ego_list.append(
                get_always_no_collision_between_agents_3d_predicate_joint(
                    ego_ind=ego_i,
                    opp_ind=ego_j,
                )
            )
    all_avoid_ego = ego_avoid_ego_list[0]
    for j in range(1, len(ego_avoid_ego_list)):
        all_avoid_ego = And(all_avoid_ego, ego_avoid_ego_list[j])

    # all ego must always avoid collision with opp
    ego_avoid_opp_list = []
    for ego_i in range(num_ego):
        for opp_i in range(num_opp):
            ego_avoid_opp_list.append(
                get_always_no_collision_between_agents_3d_predicate_joint(
                    ego_ind=ego_i,
                    opp_ind=num_ego + opp_i,
                )
            )
    all_avoid_opp = ego_avoid_opp_list[0]
    for j in range(1, len(ego_avoid_opp_list)):
        all_avoid_opp = And(all_avoid_opp, ego_avoid_opp_list[j])

    # all ego must always satisfy altitude rules
    ego_alti_list = []
    for ego_i in range(num_ego):
        egoalt_1 = get_altitude_rule_3d_predicate_joint(
            zone_bounds=zone1,
            altitude_bounds=altitude1,
            agent_ind=ego_i,
            x_ind=0,
            z_ind=2,
        )
        egoalt_2 = get_altitude_rule_3d_predicate_joint(
            zone_bounds=zone2,
            altitude_bounds=altitude2,
            agent_ind=ego_i,
            x_ind=0,
            z_ind=2,
        )

        ego_alti_list.append((egoalt_1 & egoalt_2))
    all_ego_alti = ego_alti_list[0]
    for j in range(1, len(ego_alti_list)):
        all_ego_alti = And(all_ego_alti, ego_alti_list[j])

    # --------------OPP--------------------
    # all opp must always try to collide with ego
    opp_crash_ego_list = []
    for opp_i in range(num_opp):
        for ego_i in range(num_ego):
            opp_crash_ego_list.append(
                get_collision_between_agents_3d_predicate_joint(
                    ego_ind=ego_i,
                    opp_ind=num_ego + opp_i,
                )
            )
    all_crash_ego = opp_crash_ego_list[0]
    for j in range(1, len(opp_crash_ego_list)):
        all_crash_ego = Or(all_crash_ego, opp_crash_ego_list[j])
    # all opp must always satisfy altitude rules
    opp_alti_list = []
    for opp_i in range(num_opp):
        oppalt_1 = get_altitude_rule_3d_predicate_joint(
            zone_bounds=zone1,
            altitude_bounds=altitude1,
            agent_ind=num_ego + opp_i,
            x_ind=0,
            z_ind=2,
        )
        oppalt_2 = get_altitude_rule_3d_predicate_joint(
            zone_bounds=zone2,
            altitude_bounds=altitude2,
            agent_ind=num_ego + opp_i,
            x_ind=0,
            z_ind=2,
        )
        opp_alti_list.append((oppalt_1 & oppalt_2))
    all_opp_alti = opp_alti_list[0]
    for j in range(1, len(opp_alti_list)):
        all_opp_alti = And(all_opp_alti, opp_alti_list[j])

    # return formulas
    ego_formula = (
        all_goal_reach & all_avoid_obs & all_avoid_ego & all_avoid_opp & all_ego_alti
    )
    opp_formula = all_crash_ego & all_opp_alti
    return ego_formula, opp_formula


def get_reach_avoid_two_team_formula_cfmjx(
    obs,
    goal,
    T,
    num_ego,
    num_opp,
    min_z=0.05,
    max_z=3.0,
):
    """
    Gets reach-avoid STL formulas for two-team drone game in CFMJX environment.

    Args:
        obs: array of shape (n, 6), each row is (x1, x2, y1, y2, z1, z2) defining an obstacle box
        goal: array of shape (m, 6), each row is (x1, x2, y1, y2, z1, z2) defining a goal box
        T: int, time horizon
        num_ego: int, number of ego agents
        num_opp: int, number of opponent agents
        min_z: float, minimum altitude for agents
        max_z: float, maximum altitude for agents
    
    Returns:
        ego_formula: STL formula for ego team
        opp_formula: STL formula for opponent team
    """
    # trajectory shape should be (num_ego + num_opp, time_dim, state_dim)
    # --------------EGO--------------------
    # for ego, disjunction of all ego agents inside each box, then conjunction of all boxes
    goal_reach_list = []
    for goal_i in range(goal.shape[0]):
        inside_list = []
        for ego_i in range(num_ego):
            inside_list.append(
                get_inside_box_3d_predicate_joint(goal[goal_i], agent_ind=ego_i)
            )

        inside_goal = inside_list[0]
        for i in range(1, len(inside_list)):
            inside_goal = Or(inside_goal, inside_list[i])

        goal_reach_list.append(Eventually(inside_goal, interval=[int(T / 2), T]))
        # goal_reach_list.append(Eventually(inside_goal))
    all_goal_reach = goal_reach_list[0]
    for j in range(1, len(goal_reach_list)):
        all_goal_reach = And(all_goal_reach, goal_reach_list[j])

    # all ego must always avoid obs
    avoid_obs_list = []
    for obs_i in range(obs.shape[0]):
        for ego_i in range(num_ego):
            avoid_obs_list.append(
                get_always_outside_box_3d_predicate_joint(obs[obs_i], agent_ind=ego_i)
            )
    all_avoid_obs = avoid_obs_list[0]
    for j in range(1, len(avoid_obs_list)):
        all_avoid_obs = And(all_avoid_obs, avoid_obs_list[j])

    # all ego must always avoid collision with each other
    ego_avoid_ego_list = []
    for ego_i in range(num_ego):
        for ego_j in range(ego_i + 1, num_ego):
            ego_avoid_ego_list.append(
                get_always_no_collision_between_agents_3d_predicate_joint(
                    ego_ind=ego_i,
                    opp_ind=ego_j,
                )
            )
    all_avoid_ego = ego_avoid_ego_list[0]
    for j in range(1, len(ego_avoid_ego_list)):
        all_avoid_ego = And(all_avoid_ego, ego_avoid_ego_list[j])

    # all ego must always avoid collision with opp
    ego_avoid_opp_list = []
    for ego_i in range(num_ego):
        for opp_i in range(num_opp):
            ego_avoid_opp_list.append(
                get_always_no_collision_between_agents_3d_predicate_joint(
                    ego_ind=ego_i,
                    opp_ind=num_ego + opp_i,
                )
            )
    all_avoid_opp = ego_avoid_opp_list[0]
    for j in range(1, len(ego_avoid_opp_list)):
        all_avoid_opp = And(all_avoid_opp, ego_avoid_opp_list[j])

    # all ego must always be above min_z and below max_z
    ego_alti_list = []
    for ego_i in range(num_ego):
        egoalt_1 = get_altitude_rule_3d_predicate_cfmjx(
            min_z=min_z, max_z=max_z, agent_ind=ego_i, z_ind=2
        )

        ego_alti_list.append((egoalt_1))
    all_ego_alti = ego_alti_list[0]
    for j in range(1, len(ego_alti_list)):
        all_ego_alti = And(all_ego_alti, ego_alti_list[j])

    # --------------OPP--------------------
    # all opp must always try to collide with ego
    opp_crash_ego_list = []
    for opp_i in range(num_opp):
        for ego_i in range(num_ego):
            opp_crash_ego_list.append(
                get_collision_between_agents_3d_predicate_joint(
                    ego_ind=ego_i,
                    opp_ind=num_ego + opp_i,
                )
            )
    all_crash_ego = opp_crash_ego_list[0]
    for j in range(1, len(opp_crash_ego_list)):
        all_crash_ego = Or(all_crash_ego, opp_crash_ego_list[j])
    # all opp must always satisfy altitude rules
    opp_alti_list = []
    for opp_i in range(num_opp):
        oppalt_1 = get_altitude_rule_3d_predicate_cfmjx(
            min_z=min_z, max_z=max_z, agent_ind=num_ego + opp_i, z_ind=2
        )
        opp_alti_list.append(oppalt_1)
    all_opp_alti = opp_alti_list[0]
    for j in range(1, len(opp_alti_list)):
        all_opp_alti = And(all_opp_alti, opp_alti_list[j])

    # return formulas
    ego_formula = (
        all_goal_reach & all_avoid_obs & all_avoid_ego & all_avoid_opp & all_ego_alti
    )
    opp_formula = all_crash_ego & all_opp_alti
    return ego_formula, opp_formula
