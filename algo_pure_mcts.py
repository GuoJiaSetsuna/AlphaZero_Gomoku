import numpy as np
import copy
from operator import itemgetter

from node import TreeNode


class PureMCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, c_puct=5, n_simulate=10000):
      """
      c_puct: a number in (0, inf) that controls how quickly exploration
          converges to the maximum-value policy. A higher value means
          relying on the prior more.
      """
      self._root = TreeNode(None, 1.0)
      self._c_puct = c_puct
      self._n_simulate = n_simulate
    pass

    def _get_weight_of_available_move(self, board):
      """
      _policy: a function that takes in a board state and outputs
      a list of (action, probability) tuples and also a score in [-1, 1]
      (i.e. the expected value of the end game score from the current
      player's perspective) for the current player.

      policy is a function that takes in a state and outputs a list of (action, probability)
      tuples and a score for the state"""

      # return uniform distribution as weight for pure MCTS
      action_weights = np.ones(len(board.availables))/len(board.availables)
      return zip(board.availables, action_weights)
    pass

    # 針對某個leaf，深度擴展一層leaf
    # 1. 先走到leaf
    # 2. 針對這個leaf，再深度擴展一層leaf
    def _simulate(self, state):
      """
      Run a single simulate from the root to the leaf 
      , getting a value at the leaf and propagating it back through its parents.
      State is modified in-place, so a copy must be provided.
      """

      # 1. 先走到leaf
      node = self._root
      while(1):
        if node.is_leaf():
            break
        pass
        # Greedily select next move.
        action, node = node.select(self._c_puct)
        state.do_move(action)
      pass

      # Check for end of game
      end, winner = state.game_end()
      if not end:
        # 2. 針對目前盤面的可下子位置，每個位置配置uniform distribution的權重
        action_weights = self._get_weight_of_available_move(state)
        # 3. 針對這個leaf，再深度擴展一層leaf，
        #    這個leaf的prior probability設定為action_probs
        node.expand(action_weights)
      pass

      # 4. 針對這個棋盤，隨機模擬玩一場遊戲值到決定勝負，並取得贏家是誰 (win_trend)
      win_trend = self._simulate_play_one_full_game(state)

      # 5. 從leaf開始回頭更新每個節點的數據，一直更新到root
      node.back_propagation(-win_trend)
    pass

    # uniform distribution 隨機玩完一盤局
    def _simulate_play_one_full_game(self, state, limit=1000):
      """Use the _simulate_play_onf_full_game to play until the end of the game,
      returning +1 if the current player wins, -1 if the opponent wins,
      and 0 if it is a tie.
      """
      player = state.get_current_player()
      for i in range(limit):
        end, winner = state.game_end()
        if end:
          break
        pass

        ###########
        # 隨機挑一步 (uniform distribution)

        # 1. 先建立一個大小為len(state.availables)的陣列
        #    陣列裡每一個元素為隨機值: (0.0~1.0)
        _action_probs_tmp = np.random.rand(len(state.availables))

        # 2. 建立一個map: [(位置, 隨機值),(位置, 隨機值),...,(位置, 隨機值)]
        action_probs = zip(state.availables, _action_probs_tmp)

        # 3. 有最大隨機值的(位置, 隨機值)配對
        max_action, random_value = max(action_probs, key=itemgetter(1))

        # 4. 走一步
        state.do_move(max_action)
      else:
        # If no break from the loop, issue a warning.
        print("WARNING: _simulate_play_one_full_game reached move limit")
      pass


      if winner == -1:  # tie
        return 0
      else:
        return 1 if winner == player else -1
      pass
    pass

    def _get_move(self, state):
      # 將【造訪節點次數 visit】作為【權重】
      # 找出造訪次數最多的move，這個move就是最好的move
      # 跟人類一樣，思考最多次的決策，應該就是最好的決策


      for n in range(self._n_simulate):
        state_copy = copy.deepcopy(state)
        self._simulate(state_copy)
      pass

      suggested_move = max(self._root._children.items(),
                       key=lambda act_node: act_node[1]._n_visits)[0]

      return suggested_move
    pass

    def reset_decision_tree(self):
      self._root = TreeNode(None, 1.0)
    pass

    def __str__(self):
        return "PureMCTS"
    pass
pass

class PureMCTSPlayer(object):
  """AI player based on MCTS"""
  def __init__(self, c_puct=5, n_simulate=2000):
    self.c_puct = 5
    self.n_simulate = n_simulate
    self.brain = PureMCTS( self.c_puct, self.n_simulate)
    # log_str
    self.log_str          = f"MCTS_Pure_{n_simulate}"
  pass

  def set_player_ind(self, p):
    self.player = p
  pass

  def set_n_simulate(self, i_n_simulate):
    self.brain._n_simulate = i_n_simulate
  pass

  def get_action(self, board):

    sensible_moves = board.availables

    if len(sensible_moves) > 0:

      move = self.brain._get_move(board)

      self.brain.reset_decision_tree()

      # 回傳使用這個move的原因
      _log_of_move_probs = None

      return move, _log_of_move_probs

    else:
      print("WARNING: the board is full")
    pass

  pass

  def __str__(self):
      return "PureMCTS {}".format(self.player)
  pass
pass