import os

NUM_TRAINING_EPISODES = 1000000


# main training loop
for i in range(0, NUM_TRAINING_EPISODES):
    os.system('./cli/battlesnake play -W 11 -H 11 --name Greg --url http://0.0.0.0:8000 -g solo')
