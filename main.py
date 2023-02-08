# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import typing

from helpers import get_state
from agent import Agent

state_for_dead = 0


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "al20vortex",
        "color": "#0096FF",
        "head": "caffeine",
        "tail": "round-bum",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    return


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    agent.collect_experience(state_for_dead, game_state, True)


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    global state_for_dead
    state = get_state(game_state)
    state_for_dead = state
    agent.collect_experience(state, game_state, False)
    next_move = agent.get_next_move(state)
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    agent = Agent()
    run_server({"info": info, "start": start, "move": move, "end": end})
