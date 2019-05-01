import argparse
import tensorflow as tf

def build_summaries(n):
    # Define parameters to be logged
    rewards = [tf.Variable(0.) for i in range(n)]
    loss = [tf.Variable(0.) for i in range(n)]

    for i in range(n):
        tf.summary.scalar("Reward" + str(i), rewards[i])

    summary_vars = rewards
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--random-seed", type=int, default=123, help="Random Seed")

    # Model size parameters
    parser.add_argument("--critic-units", type=int, default=1024, help="number of units in the mlp")
    parser.add_argument("--enc-units", type=int, default=512, help="number of units in the mlp")
    parser.add_argument("--readProj", type=int, default=200, help="number of units in the mlp")
    parser.add_argument("--writeProj", type=int, default=200, help="number of units in the mlp")
    parser.add_argument("--action-units", type=int, default=256, help="number of units in the mlp")
    parser.add_argument("--query_units", type=int, default=128, help="number of control units for reasoning")
    parser.add_argument("--memUnits", type=int, default=200, help="number of memory units for reasoning")
    parser.add_argument("--att_units", type=int, default=128, help="dimension of pre-attention interactions space")


    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="learning rate for MADDPG Actor Adam optimizer")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="learning rate for MADDPG Critic  Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")

    """"----------Dropouts----------"""
    parser.add_argument("--memory_dropout",  default = 0.85, type = float,    help = "dropout on the recurrent memory")
    parser.add_argument("--read_dropout",    default = 0.85, type = float,    help = "dropout of the read unit")
    parser.add_argument("--write_dropout",   default = 1.0, type = float,    help = "dropout of the write unit")
    parser.add_argument("--output_dropout",  default = 0.85, type = float,   help = "dropout of the output unit")

    ## nonlinearities
    parser.add_argument("--relu", default="STD", choices=["STD", "PRM", "ELU", "LKY", "SELU"], type=str, help="type of ReLU to use: standard, parametric, ELU, or leaky")
    # parser.add_argument("--reluAlpha",  default = 0.2, type = float,    help = "alpha value for the leaky ReLU")

    parser.add_argument("--mulBias", default=0.0, type=float, help="bias to add in multiplications (x + b) * (y + b) for better training")  #

    parser.add_argument("--imageLinPool", default=2, type=int, help="pooling for image linearizion")

    """Call Parameters"""
    parser.add_argument("---queryInputAct", type=str, default="NON", choices=["NON", "RELU", "TANH"], help="Activation function for query in to the call")

    """Action Calculation"""
    parser.add_argument("---feedMemObsAction", action="store_true", default=True, help="Feed memory and observation for action projection")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="MD-MADDPG", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="benchmark_files", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="learning_curves", help="directory where plot data is saved")
    return parser.parse_args()
