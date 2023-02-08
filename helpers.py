import numpy as np
import torch

prev_fruit_board = None

device = torch.device('mps')
print(f"Using {device} device")

def get_state(game_state) -> torch.Tensor:
    """
    Represent the entire board with a different value for each depending
        on what is there (head, body, fruit)
    - So that would be 3 different neurons for each board tile
    - so that's 363 inputs
    - let all 0 = empty
    - let first neuron 1 = head present
    - let second neuron 1 = body present
    - let third neuron 1 = fruit present
    so then we will make 3 different boards, flatten them and make our input tensor.
    """
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    # make blank boards
    head_board = np.zeros((board_height, board_width))
    body_board = np.zeros((board_height, board_width))
    fruit_board = np.zeros((board_height, board_width))

    # make head_board
    head = game_state['you']['head']
    head_board[head['x'], head['y']] = 1

    # make body_board
    body = game_state['you']['body']
    for segment in body:
        body_board[segment['x'], segment['y']] = 1

    # make fruit_board
    fruits = game_state['board']['food']
    for fruit in fruits:
        fruit_board[fruit['x'], fruit['y']] = 1

    # concatenated_arrays = np.array([head_board, body_board, fruit_board])
    # return torch.tensor(concatenated_arrays).float().reshape(1, 3, board_width, board_height)
    concatenated_arrays = np.array([head_board, body_board], dtype=np.float)
    return torch.tensor(concatenated_arrays, device=device, dtype=torch.float).reshape(1, 2, board_width, board_height)


def get_rewards_from_game_state(game_state, dead):
    global prev_fruit_board
    if dead:
        prev_fruit_board = None
        return -0.1
    else:
        reward = 0.1
        fruit_board = game_state['board']['food']
        # if prev_fruit_board and prev_fruit_board != fruit_board:
        #     if len(fruit_board) <= len(prev_fruit_board):
        #         reward += 0.9
        prev_fruit_board = fruit_board
        return reward


def get_safe_moves_state(game_state):
    # up down left right
    safe_moves = torch.tensor([1, 1, 1, 1])
    head = game_state["you"]["body"][0]
    neck = game_state["you"]["body"][1]

    if neck["x"] < head["x"]:  # Neck is left of head, don't move left
        safe_moves[2] = 0

    elif neck["x"] > head["x"]:  # Neck is right of head, don't move right
        safe_moves[3] = 0

    elif neck["y"] < head["y"]:  # Neck is below head, don't move down
        safe_moves[1] = 0

    elif neck["y"] > head["y"]:  # Neck is above head, don't move up
        safe_moves[0] = 0

    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    head = game_state['you']['head']
    if head['x'] == 0:
        safe_moves[2] = False
    if head['x'] == board_width - 1:
        safe_moves[3] = False
    if head['y'] == 0:
        safe_moves[1] = False
    if head['y'] == board_height - 1:
        safe_moves[0] = False

    # my_body will be an array of {x,y} objects
    body = game_state['you']['body']
    for segment in body:
        if segment['x'] == head['x']:
            if head["y"] == segment["y"] - 1:
                safe_moves[0] = 0
            if head["y"] == segment["y"] + 1:
                safe_moves[1] = 0
        if segment["y"] == head["y"]:
            if head["x"] == segment["x"] - 1:
                safe_moves[3] = 0
            if head["x"] == segment["x"] + 1:
                safe_moves[2] = 0

     # woo!
    return safe_moves.half().reshape(1, 4)
