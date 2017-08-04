import numpy as np
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.plotter import plotter
from rllab.spaces import Box
import matplotlib.pyplot as plt


# INPUT: data, a dictionary of arrays, with the first dimension of each array
#              corresponding to different data entries (assumes all have the
#              same length in dimension 0)
#        percent_train, the fraction of data to keep in the train set
# OUTPUT: (train_data, val_data), dictionaries in the same format as data
#         note: also shuffles the data
def split_train_val(data, percent_train=0.7):
    train_data = {}
    val_data = {}
    N = len(next(iter(data.values())))
    shuffled_idx = np.random.permutation(N)
    cut_pt = int(np.floor(percent_train*N))
    train_idx = shuffled_idx[:cut_pt]
    val_idx = shuffled_idx[cut_pt:]

    for key,val in data.items():
        train_data[key] = val[train_idx]
        val_data[key] = val[val_idx]

    return (train_data, val_data)


# INPUT: env, the OpenAI Gym environment to prob
#        x_low, x_high, arrays containing the bounds on the states to probe
#        u_low, u_high, arrays containing the bounds on the action to probe
#        N, number of samples to generate
# OUTPUT: transitions, a dictionary containing the data
# ASSUMPTIONS: env has a self.state variable, which is equivalent to the
#              observation returned by self.step()
def sample_transitions(env, x_low, x_high, u_low, u_high, N):
    transitions = {
        "x":      [],
        "u":      [],
        "x_next": []
    }
    for i in range(N):
        x = np.random.uniform(x_low, x_high)
        u = np.random.uniform(u_low, u_high)

        env.reset()
        env.env.state = x
        x_next, _, _, _ = env.step(u)

        transitions["x"].append(x)
        transitions["u"].append(u)
        transitions["x_next"].append(x_next)

    return { key:np.array(val) for key,val in transitions.items() }


# INPUT: env_name, the string corresponding to an OpenAI Gym environment
#        n_itr, the number of rl iterations to perform when capturing transitions
#        plot, whether or not render the environment when capturing trajectories
# OUTPUT: transitions, a dictionary containing the data
def sample_transitions_rl(env_name, n_itr, plot=False):
    transitions = {
        "x":      [],
        "u":      [],
        "x_next": []
    }

    if plot:
        plotter.init_worker()

    env = normalize(GymEnv(env_name))

    x_dim = env.spec.observation_space.flat_dim
    u_dim = env.spec.action_space.flat_dim

    def log_transitions(paths):
        for path in paths:
            transitions["x"].append( path["observations"][:-1] )
            transitions["u"].append( path["actions"][:-1] )
            transitions["x_next"].append( path["observations"][1:] )

    env.log_diagnostics = log_transitions

    if isinstance(env.spec.action_space, Box):
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(8, 8)
        )
    else:
        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(8, 8)
        )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=n_itr,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        plot=plot,
        store_paths=True,

    )
    algo.train()
    transitions = { key:np.concatenate(val) for key,val in transitions.items() }
    return transitions


color_list = ['#00697D','#70007D','#338030']
facecolor_list = ['#00DDFF','#DD00FF','#66FF66']

# Generate plots of function over a slice of the state space
def sample_and_plot_results(env, pdm_list, x_range, u_range, N=100, labels=None):
    # determine which value to use as the x axis
    key = None
    dim = -1
    x_diff = (x_range[1,:] != x_range[0,:])
    x_dim = np.nonzero(x_diff)[0]

    u_diff = (u_range[1,:] != u_range[0,:])
    u_dim = np.nonzero(u_diff)[0]
    if len(x_dim) + len(u_dim) != 1:
        raise Exception('Input range must only vary in one dimension!')

    if len(x_dim) == 1:
        dim = x_dim[0]
        key = "x"
    if len(u_dim) == 1:
        dim = u_dim[0]
        key = "u"

    true_transitions = sample_transitions(env, x_range[0,:], x_range[1,:], u_range[0,:], u_range[1,:], N)
    x = true_transitions[key][:,dim]
    y = true_transitions["x_next"] - true_transitions["x"]

    # sort data by x axis
    sorted_i = x.argsort()

    # Now, use the PDMs to make predictions over x
    y_hat = []
    y_plus1 = []
    y_plus2 = []
    y_sub1 = []
    y_sub2 = []
    for pdm in pdm_list:
        predictions = pdm.predict(true_transitions["x"], true_transitions["u"])

        full_sd = np.sqrt(predictions["epistemic_unc_of_mean"]) #predictions["aleatoric_unc"] +

        y_hat.append(  predictions["x"] - true_transitions["x"] )
        y_plus1.append(  predictions["x"] - true_transitions["x"] + 1*full_sd )
        y_sub1.append( predictions["x"] - true_transitions["x"] - 1*full_sd )
        y_plus2.append(  predictions["x"] - true_transitions["x"] + 2*full_sd )
        y_sub2.append(  predictions["x"] - true_transitions["x"] - 2*full_sd )

    if not labels:
        labels = [str(i+1) for i in range(len(pdm_list))]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(8, 6), dpi=80)
    plt.hold(1)

    K = y.shape[1]
    for k in range(K):
        plt.subplot(K*100 + 10 + k + 1)
        legend_handles = []
        for i in range(len(pdm_list)):
            handle, = plt.plot(x[sorted_i], y_hat[i][sorted_i,k], color=color_list[i], label=labels[i], linewidth=2)
            plt.fill_between(x[sorted_i], y_sub1[i][sorted_i,k], y_plus1[i][sorted_i,k],
                                alpha=0.2, edgecolor=color_list[i], facecolor=facecolor_list[i])
            plt.fill_between(x[sorted_i], y_sub2[i][sorted_i,k], y_plus2[i][sorted_i,k],
                                alpha=0.2, edgecolor=color_list[i], facecolor=facecolor_list[i])

            legend_handles.append(handle)

        handle, = plt.plot(x[sorted_i], y[sorted_i,k], color='#000000', label='Ground Truth', linewidth=2)
        legend_handles.append(handle)

        if k == 0:
            plt.legend(handles=legend_handles, fontsize=11, loc=4)
        plt.ylabel(r"$\mathbf{x}_" + str(k) + r"(t+1) - \mathbf{x}_" + str(k) + r"(t)$", fontsize=16)
        if k == K - 1:
            plt.xlabel(r"$\mathbf{" + key + r"}_" + str(dim) + r"(t)$", fontsize=16)
        plt.grid(True)


    plt.show()
