
import argparse
import tensorflow as tf

# ----------------------------------------
# Global variables
arg_lists = []
parser = argparse.ArgumentParser()
# we want to focus on laerning stability
# increase the discount value

# exp_main

filename = "exp_main"

# # Possible actions
# shoot = [1, 0, 0]
# left = [0, 1, 0]
# right = [0, 0, 1]

# ----------------------------------------
# Macro for arparse
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def get_actions():
    actions = []
    # m_left_right = [[True, False], [False, True], [False, False]]  # move left and move right
    # attack = [[True], [False]]
    # m_forward_backward = [[True, False], [False, True], [False, False]]  # move forward and backward
    move_with_speed = [[True, False], [True, True], [False, False]]
    t_left_right = [[True, False], [False, True], [False, False]]  # turn left and turn right

    # for i in attack:
    for j in move_with_speed:
        # for k in m_forward_backward:
            for l in t_left_right:
                actions.append(j+l)
    return actions

actions=get_actions()

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-5,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--gradient_clip_val", type=float,
                       default=10,
                       help="Amount to clip gradients by (stabilisation)")

train_arg.add_argument("--living_reward", type=float,
                       default=-0.001,
                       help="Living reward for the agent")

train_arg.add_argument("--episode_timeout_steps", type=float,
                       default=2000,
                       help="Steps taken by agent until timeout")

train_arg.add_argument("--discount", type=float,
                       default=0.99,
                       help="Ensures Q function will converge by providing diminishing returns. Must be < 1")

train_arg.add_argument("--epsilon", type=float,
                       default=1.0,
                       help="Probability of e-greedy exploration. Reduced linearly over time")

train_arg.add_argument("--epsilon_decay_rate", type=int,
                       default=0.9995,
                       help="The epsilon decay rate multiplier after each episode")

train_arg.add_argument("--batch_size", type=int,
                       default=64,
                       help="Number of experiences to sample from memory during training")

train_arg.add_argument("--episodes", type=int,
                       default=5100,
                       help="Number of episodes to train on")

train_arg.add_argument("--entropy_rate", type=int,
                       default=1e-4,
                       help="Ratio of entropy regularization to apply to loss")

train_arg.add_argument("--update_target_rate", type=int,
                       default=0.25,
                       help="Frequency to update target network. 1.0 is every episode, 0.1 is every 10 episodes, etc...")

train_arg.add_argument("--alpha", type=float,
                       default=0.6,
                       help="Exponent for experience replay probability (0 is uniform dist)")

train_arg.add_argument("--temp", type=int,
                       default=0.99,
                       help="Temperature for boltzmann exploration (higher = more exploration)")

train_arg.add_argument("--log_dir", type=str,
                       default=f"./{filename}/",
                       help="Directory to save logs")

train_arg.add_argument("--log_freq", type=int,
                       default=10,
                       help="Number of steps before logging weights")

train_arg.add_argument("--save_dir", type=str,
                       default="./saves/",
                       help="Directory to save current model")

train_arg.add_argument("--save_freq", type=int,
                       default=10000,
                       help="Number of episodes before saving model")

train_arg.add_argument("-f", "--extension", type=str,
                       default=None,
                       help="Specific name to save training session or restore from")

# ----------------------------------------
# Arguments for testing
test_arg = add_argument_group("Testing")

test_arg.add_argument("--test_episodes", type=int,
                       default=1,
                       help="Number of episodes to test on")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--model", type=str,
                       default="atari",
                       choices=["atari"],
                       help="CNN architecture to use")

model_arg.add_argument("--activ", type=str,
                       default="relu",
                       choices=["relu", "elu", "selu", "tanh", "sigmoid"],
                       help="Activation function to use")

model_arg.add_argument("--init", type=str,
                       default="glorot_normal",
                       choices=["glorot_normal", "glorot_uniform", "random_normal", "random_uniform", "truncated_normal"],
                       help="Initialization function to use")

model_arg.add_argument("--actions", type=int,
                       default=actions,
                       help="Possible actions to take")

model_arg.add_argument("--skiprate", type=int,
                       default=4,
                       help="Number of frames to skip during each action. Current action will be repeated for duration of skip")

model_arg.add_argument("--num_frames", type=int,
                       default=4,
                       help="Number of stacked frames to send to CNN, depicting motion")

# ----------------------------------------
# Arguments for memory
mem_arg = add_argument_group("Memory")

mem_arg.add_argument("--cap", type=int,
                       default=100000,
                       help="Maximum number of transitions in replay memory")

# ----------------------------------------
# Function to be called externally
def get_config():
    config, unparsed = parser.parse_known_args()

    # If there are unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        parser.print_usage()
        exit(1)

    return config
