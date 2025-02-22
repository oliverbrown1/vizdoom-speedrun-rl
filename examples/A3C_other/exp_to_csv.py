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

#df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
# df.to_csv("output.csv")
# Load the TensorBoard logs
# logdir = './files/summaries'
# logs = tf.data.experimental.load(logdir)

# Convert to DataFrame
# df = tf.data.experimental.to_dataframe(logs)



# mwh_10 -> gamma = 0.97
# mwh_11 -> lr=1e-6, gamma = 0.99, episodes = 2000
# mwh_12 -> lr=1e-4, episodes = 2000
filename = "mwh_17"

df=tflog2pandas(f"./{filename}/summaries/agent_0")
# Save to CSV
df.to_csv(f"./results/{filename}.csv")
# df.to_csv(f'./results/{filename}.csv', index=False)

parameters = "actions=FORWARD, SPEED, TLEFT, TRIGHT\n"
parameters += "action_combinations=all\n"
parameters += "max_episodes=5000\n"
parameters += "num_workers=8\n"
parameters += "living_reward=-0.001\n"
parameters += "gamma=0.99\n"
parameters += "lr=1e-6\n"
parameters += "frameskip=4\n"
parameters += "network_update_interval= max steps"
parameters += "gradient_clip_val=30"



with open(f"./results/{filename}.txt", "w") as file:
    file.write(parameters)
