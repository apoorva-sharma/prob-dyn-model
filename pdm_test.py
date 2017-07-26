from ProbabilisticDynamicsModel import *
from utils import *

import gym
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Probabilistic Dynamics Model")
    parser.add_argument("--env", default="MountainCarContinuous-v0", help="the OpenAI gym environment to use.")
    parser.add_argument("--layer_sizes", type=int, nargs="+", metavar='SIZE', default=[64,64,64], help="Sizes of the hidden layers in the PDM")
    parser.add_argument("--lr", default=2e-3, type=float, help="Learning rate to use when training the PDM")
    parser.add_argument("--label", "-l", default="pdm_test", help="Filename to use when saving training logs")
    parser.add_argument("--n_itr", default=10, type=int, help="Number of TRPO iterations on the env to gather data from")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability to use in the PDM")
    parser.add_argument("--n_epochs", default=5, type=int, help="Number of epochs to use when training PDM")
    parser.add_argument("--num_monte_carlo", default=50, type=int, help="Number of MC samples to use for epistemic uncertainty estimation")
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size when training PDM")
    parser.add_argument("--plot", action="store_true", help="Whether to render trajectories")
    parser.add_argument("--validation_curves", action="store_true", help="Whether to compute metrics on validation data while training")
    parser.add_argument("--fake_data", action="store_true", help="Whether to randomly sample transitions rather than doing RL")
    parser.add_argument("--x_low", type=float, nargs="+", metavar='VAL', default=[-0.5, -0.2], help="Lower bound on states to sample (only works with fake_data)")
    parser.add_argument("--x_high", type=float, nargs="+", metavar='VAL', default=[0.0, 0.2], help="Upper bound on states")
    parser.add_argument("--u_low", type=float, nargs="+", metavar='VAL', default=[-0.5], help="Lower bound on action")
    parser.add_argument("--u_high", type=float, nargs="+", metavar='VAL', default=[0.5], help="Upper bound on action")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of transitions to sample")

    args = parser.parse_args()
    print("Running with the following settings:\n", args)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    env_name = args.env

    if not args.fake_data:
        transitions = sample_transitions_rl(env_name, args.n_itr, args.plot)
    else:
        env = gym.make(args.env)
        transitions = sample_transitions(env, args.x_low, args.x_high, args.u_low, args.u_high, args.num_samples)

    x_dim = transitions["x"].shape[1]
    u_dim = transitions["u"].shape[1]

    # Now train the PDM on the observed transitions
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        pdm = MLPDynamicsModel(sess, x_dim, u_dim,
                               hidden_layer_sizes=args.layer_sizes,
                               dropout_prob=args.dropout,
                               num_mc_samples=args.num_monte_carlo,
                               filename=args.label,
                               writer_path=env_name)
        pdm.build_model()
        cfg = MLP_DM_cfg
        cfg["lr"] = args.lr
        cfg["batch_size"] = args.batch_size
        cfg["n_epochs"] = args.n_epochs
        cfg["store_val"] = args.validation_curves
        pdm.train(transitions, cfg)

        # test_transitions = sample_transitions(env, [-1,0.1], [1,0.1], [0.0], [0.0], 100)
        # predictions = pdm.predict(test_transitions["x"], test_transitions["u"])
        x_range = np.array([[-1, 0.1],[1, 0.1]])
        u_range = np.array([[0.0], [0.0]])
        sample_and_plot_results(gym.make(args.env), pdm, x_range, u_range)
