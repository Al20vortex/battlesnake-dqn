import os

NUM_TRAINING_EPISODES = 1000000


# main training loop
for i in range(0, NUM_TRAINING_EPISODES):
    os.system('./cli/battlesnake play -W 5 -H 5 --name snek --url http://0.0.0.0:8000 -g solo')
