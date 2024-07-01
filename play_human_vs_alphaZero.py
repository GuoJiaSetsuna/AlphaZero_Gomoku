from __future__ import print_function
import pickle

import random
import numpy as np
import os
import sys
import datetime
import time

# Loading Config 載入設定檔，本程式中大部分的變數設定都存於config.py
if os.path.exists('config.py'):
  from config import *
pass
# 將其他檔案作為函式庫匯入(game.py、algo_alphagoZero.py、algo_human.py)
from game import Board, Game
from algo_alphaZero import AlphaZero_Player as ALGO_AlphaZero
from algo_human import Algo_Human
from elo_rating import EloRating as ELO 
# 建立啟動函式
def run():
  # 從config取出勝利子數(n)、棋盤大小(width, height)與AI模型位置(model_file)
  n = PLAY_N_IN_ROW
  width, height = PLAY_BOARD_WIDTH, PLAY_BOARD_HEIGHT
  model_file = PLAY_MODEL_PATH
  # 以取出的設定結合game.py函式建立棋盤與遊戲
  board = Board(width=width, height=height, n_in_row=n)
  game = Game(board)

  # 以algo_alphagoZero.py函式建立AlphaGo Zero玩家
  alphaZero_player = ALGO_AlphaZero(    c_puct=PLAY_C_PUCT,
                           n_playout=PLAY_ALPHAZERO_SIM_TIMES,
                           model_path=model_file
                           ) 
  # 以algo_human.py函式建立人類玩家
  # human player, input your move in the format: 2,3
  human_player = Algo_Human()
  # 以前面所建立的AI玩家與人類玩家啟動遊戲
  # set start_player=0 for human first
  # set start_player=1 for alphaZero_player first
  a = 0
  b = 1600

  # 計算程式碼執行時間的起始點
  start_time = time.time()

  # game.start_play(human_player, alphaZero_player, start_player = PLAY_START_PLAYER, is_shown = PLAY_IS_SHOWN)
  ELO(a,b,30,game.start_play(human_player, alphaZero_player, start_player = PLAY_START_PLAYER, is_shown = PLAY_IS_SHOWN))

  # 計算程式碼執行時間的結束點
  end_time = time.time()

  # 計算執行時間
  execution_time = end_time - start_time

  print("ELO function execution time:", execution_time, "seconds")  
  print(a)
  print(b)
  # game.start_play(human_player, alphaZero_player, start_player = PLAY_START_PLAYER, is_shown = PLAY_IS_SHOWN)

pass
# 執行啟動函式
if __name__ == '__main__':
  run()
pass