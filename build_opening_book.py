import random

from isolation.isolation import Isolation, DebugState, _WIDTH, _HEIGHT

from my_custom_player import CustomPlayer
from sample_players import RandomPlayer, MinimaxPlayer, GreedyPlayer

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

#export heatmap images for wins-losses per move
EXPORT_IMAGES = True

BOOK_DEPTH = 4
#number of rounds to simulate after book depth has been met if we're using win_score
NUM_ROUNDS = 10

#scalars for how own and opponent liberties affect score
OWN_SCALE = 1
OPP_SCALE = 1

#BOOK_PATH = "./own{}_opp{}_depth{}.pickle".format(OWN_SCALE, OPP_SCALE, BOOK_DEPTH)
BOOK_PATH = "./data.pickle"

def build_table(num_rounds, depth):
  #builds a mapping [state -> most optimal move]
  #the move is based on the score from get_score()
  from collections import defaultdict, Counter

  book = defaultdict(Counter)
  state = Isolation()

  build_tree(state, book, depth)

  return ({k: max(v, key=v.get) for k, v in book.items()}, book)

def build_tree(state, book, depth):
  #if we're out of depth or game is over
  if depth <= 0 or state.terminal_test():
    #return negative result of the rest of the game...?
    return -get_score(state)#-

  #for every availale action
  total_score = 0
  actions = state.actions()
  for i, action in enumerate(actions):
    #get the score for that action (recursively)
    score = build_tree(state.result(action), book, depth - 1)
    #store the score of the state, action pair
    book[state][action] += score
    #add score
    total_score += score
    #if we're at the top level, report %complete
    if depth == BOOK_DEPTH:
      print("{:.2f}%".format((i + 1) / len(actions) * 100))
  return -total_score#-

def get_score(state):
  return win_score(state)
  #return moves_score(state)

#from sample_players.py
# our liberties - opponent liberties
def moves_score(state):
  player_id = state.player()
  own_loc = state.locs[player_id]
  opp_loc = state.locs[1 - player_id]
  own_liberties = state.liberties(own_loc)
  opp_liberties = state.liberties(opp_loc)
  return OWN_SCALE * len(own_liberties) - OPP_SCALE * len(opp_liberties)

# +1 if player wins, -1 if player loses (slow..)
def win_score(start_state):
  score = 0
  for i in range(NUM_ROUNDS):
    state = start_state
    player_id = state.player()
    while not state.terminal_test():
      state = state.result(random.choice(state.actions()))
    score += -1 if state.utility(player_id) < 0 else 1
  return score

def moves2heatmap(moves):
  heatmap = np.zeros([_WIDTH, _HEIGHT])
  for move, score in moves.items():
    k = DebugState.ind2xy(move)
    heatmap[k[0], k[1]] = score

  return heatmap

def plot_heatmap(heatmap, min, max):
  ax = sns.heatmap(heatmap, cmap="Blues")
  ax.xaxis.tick_top()
  return ax

if __name__ == "__main__":
  (book, ratios) = build_table(num_rounds=NUM_ROUNDS, depth=BOOK_DEPTH)

  #print best opening move
  game_start = Isolation()
  best_opening = book[game_start]
  print("Best opening move: {}".format(DebugState.ind2xy(best_opening)))

  #print best response to opening move
  resulting_state = game_start.result(best_opening)
  best_response = book[resulting_state]
  print("Best response: {}".format(DebugState.ind2xy(best_response)))

  #build heatmap of opening move wins..
  opening_moves = ratios[game_start]

  xy = [DebugState.ind2xy(ind) for ind in opening_moves]
  (xs, ys) = zip(*xy)
  width = max(xs) + 1
  height = max(ys) + 1

  if EXPORT_IMAGES:
    heatmap = np.zeros([_WIDTH, _HEIGHT])

    for move, score in opening_moves.items():
      k = DebugState.ind2xy(move)
      heatmap[k[0], k[1]] = score

    ax = plot_heatmap(heatmap, 0, 1)
    plt.savefig("./images/initial.png")

    #print every best 2nd move
    for move, score in opening_moves.items():
      resulting_state = game_start.result(move)
      responses = ratios[resulting_state]
      heatmap = moves2heatmap(responses)
      plt.clf()
      ax = plot_heatmap(heatmap, 0, 1)
      plt.savefig("./images/{}.png".format(DebugState.ind2xy(move)))

  #save book
  pickle.dump(book, open(BOOK_PATH, "wb"))
