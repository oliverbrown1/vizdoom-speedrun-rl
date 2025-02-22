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
exp_number = 43

df=tflog2pandas(f"./files{exp_number}")
filename = f"exp_{exp_number}"


# Save to CSV
df.to_csv(f"./results/{filename}.csv")
# df.to_csv(f'./results/{filename}.csv', index=False)

parameters = "actions=FORWARD, SPEED, TLEFT, TRIGHT\n"
parameters += "lr=1e-5\n"
parameters += "living_reward=-0.001\n"
parameters += "gamma=0.99\n"
parameters += "epsilon=1\n"
parameters += "batch_size=64\n"
parameters += "max_episodes=5000\n"
parameters += "entropy_rate=1e-4\n"
parameters += "update_target_rate=0.25\n"
parameters += "alpha=0.6\n"
parameters += "frameskip=4\n"
parameters += "decay_rate=0.9995 (begins after 200 episodes)\n"
parameters += "experience_replay_size=50000"
parameters += "gradient_clip_val = 10"



with open(f"./results/{filename}.txt", "w") as file:
    file.write(parameters)
