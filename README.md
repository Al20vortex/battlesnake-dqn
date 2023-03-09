# battlesnake-dqn (WIP)


This is my work in progress attempt at making a deep q-learning agent for the Battlesnake competition. Currently the agent uses a simplified view of the world and ignores fruit, to try and survive. The agent currently can't compete with other agents, nor are there obstacles.

Todo list:
1. There's a memory leak currently preventing us from training too many iterations.
2. Overhaul how we save models to restart training from last session
3. Experiment with different models
4. Add in fruit observations to agent's 'vision', then train
5. Add walls to the game and to the agent's vision
6. Add simple enemy agents
7. Train against other deep q-learning agents
