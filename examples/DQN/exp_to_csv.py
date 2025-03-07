import tensorflow as tf
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

# exp38 -> frameskip=4, decay_rate = 0.9995, living_reward = -0.001, batch_size = 128, episodes=  1000
# exp39 -> batch_size = 64, update_target_rate = 0.25
# exp40 -> update_target_rate = 0.2, entropy_Rate = 1e-3

# mwh_1 ->  1000 episodes, replay_size = 100000
# mwh_2 -> hyperparameters found from A3C -> update_target_Rate = 0.2, episode_timeout = 3000
# files44 -> replay_size = 100k, episode_timeout_steps = 3000
# files45 -> lr=1e-6
# mwh_3 -> gradient_clip_val = 20, episode_timeout = 2000, lr=1e-6
# mwh_3 -> episode_timeout = 2000, gradient_clip_val = 10
# mwh_4 -> gradient_clip_val = 20, lr=1e-4
# mwh_5 -> lr=1e-6
# mwh_6 -> episode_timeout = 3000, gradient_clip_val = 30
# mwh_7 -> update_target_rate=  0.1
# mwh_8 -> update_target_rate = 0.2, discount = 0.97

# maze_4 -> lr=1e-4, episode_timeout = 3000
# maze_5 -> episode_timeout = 5000
# maze_6 -> episode_timeout = 4000, update_target_rate = 0.1

# maze_7 -> update_target_rate = 0.2
# maze_8 -> update_target_rate = 0.3
# maze_9 -> update_Target_Rate = 0.25, gamma = 0.95
# maze_10 -> gamma = 0.9



filename = "mwh_main"

df=tflog2pandas(f"./{filename}")


# Save to CSV
df.to_csv(f"./results/{filename}.csv")
# df.to_csv(f'./results/{filename}.csv', index=False)

parameters = "actions=FORWARD, SPEED, TLEFT, TRIGHT\n"
parameters += "lr=1e-6\n"
parameters += "living_reward=-0.001\n"
parameters += "gamma=0.99\n"
parameters += "epsilon=1\n"
parameters += "batch_size=64\n"
parameters += "max_episodes=5000\n"
parameters += "entropy_rate=1e-4\n"
parameters += "update_target_rate=0.2\n"
parameters += "alpha=0.6\n"
parameters += "frameskip=4\n"
parameters += "decay_rate=0.9995 (begins after 200 episodes)\n"
parameters += "experience_replay_size=100k\n"
parameters += "gradient_clip_val = 20\n"
parameters += "episode_timeout_steps = 2000"




with open(f"./results/{filename}.txt", "w") as file:
    file.write(parameters)
