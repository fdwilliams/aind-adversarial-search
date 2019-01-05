
from sample_players import DataPlayer

import random
import logging

#iterative deepening max depth, use None to continue until time runs out
MAX_DEPTH = None
USE_OPENING_BOOK = True
OPENING_BOOK_LENGTH = 4
#choose random moves up to this depth (for testing opening book), -1 for always random
RANDOM_PLAYS = 0

DEBUG = False

class CustomPlayer(DataPlayer):
  """ Implement your own agent to play knight's Isolation

  The get_action() method is the only required method for this project.
  You can modify the interface for get_action by adding named parameters
  with default values, but the function MUST remain compatible with the
  default interface.

  **********************************************************************
  NOTES:
  - The test cases will NOT be run on a machine with GPU access, nor be
    suitable for using any other machine learning techniques.

  - You can pass state forward to your agent on the next turn by assigning
    any pickleable object to the self.context attribute.
  **********************************************************************
  """
  def get_action(self, state):
    """ Employ an adversarial search technique to choose an action
    available in the current state calls self.queue.put(ACTION) at least

    This method must call self.queue.put(ACTION) at least once, and may
    call it as many times as you want; the caller will be responsible
    for cutting off the function after the search time limit has expired.

    See RandomPlayer and GreedyPlayer in sample_players for more examples.

    **********************************************************************
    NOTE:
    - The caller is responsible for cutting off search, so calling
      get_action() from your own code will create an infinite loop!
      Refer to (and use!) the Isolation.play() function to run games.
    **********************************************************************
    """
    if DEBUG:
      logger = logging.getLogger(__name__)
      logger.info("Begin play {}".format(state.ply_count))

    #random action:
    if RANDOM_PLAYS < 0 or state.ply_count < RANDOM_PLAYS:
      self.queue.put(random.choice(state.actions()))
      return

    #check for opening book:
    if (state.ply_count < OPENING_BOOK_LENGTH and
      USE_OPENING_BOOK and (self.data is not None)):
      if DEBUG:
        logger.info("using opening book")
      #get move from opening book
      #if state.board in self.data:
      if state in self.data:
        #move = self.data[state.board]
        move = self.data[state]
        self.queue.put(move)
        if DEBUG:
          logger.info("Move from book: {}".format(move))
        return
      elif DEBUG:
        logger.info("State {} not in book?".format(state))

    if DEBUG:
        logger.info("Using minimax")
    #iterative deepening:
    depth = 1
    while MAX_DEPTH is None or depth <= MAX_DEPTH:
      #move = self.minimax(state, depth=depth)
      move = self.alpha_beta_search(state, depth=depth)
      if DEBUG:
        logger.info("Depth {}: {}".format(depth, move))
      if move is None:
        logger.info("'None' move found; aborting search")
        logger.info(state.actions())
        return
      self.queue.put(move)
      depth += 1

  #score function from MinimaxPlayer
  def score(self, state):
    own_loc = state.locs[self.player_id]
    opp_loc = state.locs[1 - self.player_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    return len(own_liberties) - len(opp_liberties)

  #minimax w/o alpha-beta pruning
  def minimax(self, state, depth):

    def min_value(state, depth):
      if state.terminal_test(): return state.utility(self.player_id)
      #if state.terminal_test(): return self.score(state)
      if depth <= 0: return self.score(state)

      value = float("inf")
      for action in state.actions():
        value = min(value, max_value(state.result(action), depth - 1))
      return value

    def max_value(state, depth):
      if state.terminal_test(): return state.utility(self.player_id)
      #if state.terminal_test(): return self.score(state)
      if depth <= 0: return self.score(state)

      value = float("-inf")
      for action in state.actions():
        value = max(value, min_value(state.result(action), depth - 1))
      return value

    return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

  #minimax with alpha-beta pruning
  def alpha_beta_search(self, state, depth):
      """ Return the move along a branch of the game tree that
      has the best possible value.  A move is a pair of coordinates
      in (column, row) order corresponding to a legal move for
      the searching player.

      You can ignore the special case of calling this function
      from a terminal state.
      """

      def min_value(state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        #if state.terminal_test(): return self.score(state)
        if depth <= 0: return self.score(state)

        v = float("inf")
        for a in state.actions():
          v = min(v, max_value(state.result(a), alpha, beta, depth - 1))
          if v <= alpha:
            return v
          beta = min(beta, v)
        return v

      def max_value(state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        #if state.terminal_test(): return self.score(state)
        if depth <= 0: return self.score(state)

        v = float("-inf")
        for a in state.actions():
          v = max(v, min_value(state.result(a), alpha, beta, depth - 1))
          if v >= beta:
            return v
          alpha = max(alpha, v)
        return v

      alpha = float("-inf")
      beta = float("inf")
      best_score = float("-inf")
      #best_move = None
      #prevent waiting forever in run_match.py, we need to choose SOME action,
      #even if it has -inf score...
      best_move = None if len(state.actions()) == 0 else state.actions()[0]

      for a in state.actions():
        v = min_value(state.result(a), alpha, beta, depth)
        alpha = max(alpha, v)
        if v > best_score:
          best_score = v
          best_move = a
      return best_move
