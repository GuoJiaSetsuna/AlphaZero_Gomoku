from __future__ import print_function
import pickle

import random
import numpy as np
import os
import sys
import datetime

# Loading Config 載入設定檔，本程式中大部分的變數設定都存於config.py
if os.path.exists('config.py'):
  from config import *
pass
# 將其他檔案作為函式庫匯入(game.py、algo_alphagoZero.py、algo_human.py)
from game import Board, Game
from algo_alphaZero import AlphaZero_Player as ALGO_AlphaZero
from algo_pure_mcts import PureMCTSPlayer as ALGO_Pure_MCTS

# 建立啟動函式
def run():
  # 從config取出勝利子數(n)、棋盤大小(width, height)與AI模型位置(model_file)
  n = PLAY_N_IN_ROW
  width, height = PLAY_BOARD_WIDTH, PLAY_BOARD_HEIGHT
  model_file = "trained_models/best_policy_at_850_vs_mcts_10000.model"
  # 以取出的設定結合game.py函式建立棋盤與遊戲
  board = Board(width=width, height=height, n_in_row=n)
  game = Game(board)

  # 以algo_alphagoZero.py函式建立AlphaGo Zero玩家
  alphaZero_player = ALGO_AlphaZero(    c_puct=PLAY_C_PUCT,
                           n_playout=PLAY_ALPHAZERO_SIM_TIMES,
                           model_path=model_file
                           ) 
  # 以algo_pure_mcts.py函式建立純淨MCTS玩家
  # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
  mcts_player = ALGO_Pure_MCTS(c_puct = PLAY_C_PUCT, n_simulate=20000)
  # 以前面所建立的AI玩家與人類玩家啟動遊戲
  # set start_player=0 for mcts_player first
  # set start_player=1 for alphaZero_player first
  game.start_play(mcts_player, alphaZero_player, start_player = PLAY_START_PLAYER, is_shown = PLAY_IS_SHOWN)

pass
# 執行啟動函式
if __name__ == '__main__':
  run()
pass