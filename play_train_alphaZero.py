from __future__ import print_function

#import cupy as np
import numpy as np
import pandas as pd

import time
import random

import os
import sys
import datetime
import torch
import copy
import os.path
import pickle

from collections import defaultdict, deque
from tqdm import tqdm

from game import Board, Game
from algo_pure_mcts import PureMCTSPlayer as ALGO_Pure_MCTS
# CNN版本
from algo_alphaZero import AlphaZero_Player as ALGO_AlphaZero
# Self-Attention版本
from algo_alphaZero_self import AlphaZero_Player as ALGO_AlphaZero_Self

def show_time_now():
  return f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
pass

# Loading Config
if os.path.exists('config.py'):
  from config import *
else:
  raise Exception('no config.py file ')
pass

# load tool
from elo_rating import *
from db_lib import *
create_training_table()

if not os.path.exists(TRAINED_MODELS_DIR):
  os.makedirs(TRAINED_MODELS_DIR)
pass

# re-factor done
class TrainProcedure():
  def __init__(self, init_model=LOAD_MODEL_FILENAME):

    ####################################
    # training params

    # num of simulations used for the pure mcts, which is used as
    # the opponent to evaluate the trained policy
    self.pure_mcts_playout_num = DEFAULT_PURE_MCTS_PLAYOUT_NUM

    self.best_win_ratio = 0.0

    ####################################
    # AI and game setting

    self.board = Board(width    = BOARD_WIDTH,
                       height   = BOARD_HEIGHT,
                       n_in_row = N_IN_ROW)
    
    self.game = Game(self.board)


    ####################################
    # AI and game setting

    self.best_elo_0 = 0
    self.training_start_epoch = 0

    self.qkv_best_player = ALGO_AlphaZero_Self( c_puct=C_PUCT, n_simulate=N_ALPHAZERO_SIMULATE)

    self.algo_pool = []

    self.algo_pool.append(ALGO_AlphaZero_Self( c_puct=C_PUCT, n_simulate=N_ALPHAZERO_SIMULATE))
    self.algo_pool[0].elo_rate=0

    self.algo_pool.append(ALGO_Pure_MCTS(c_puct=C_PUCT, n_simulate=1000))
    self.algo_pool[1].elo_rate=2000

    self.algo_pool.append(ALGO_Pure_MCTS(c_puct=C_PUCT, n_simulate=2000))
    self.algo_pool[2].elo_rate=2000

    self.algo_pool.append(ALGO_Pure_MCTS(c_puct=C_PUCT, n_simulate=4000))
    self.algo_pool[3].elo_rate=2000

    #self.algo_pool.append(ALGO_Pure_MCTS(c_puct=C_PUCT, n_simulate=2000))
    #self.algo_pool[4].elo_rate=1600

    #self.algo_pool.append(ALGO_Pure_MCTS(c_puct=C_PUCT, n_simulate=4000))
    #self.algo_pool[5].elo_rate=1600

    # ref: https://blog.csdn.net/weixin_40522801/article/details/106563354  

    if os.path.isfile(init_model) and os.path.isfile(LOAD_LOG_FILENAME) and os.path.isfile(LOAD_BEST_MODEL_FILENAME):
      self.algo_pool[0].link_load_model(init_model)
      self.qkv_best_player.link_load_model(LOAD_BEST_MODEL_FILENAME)

      print(f"load model => {init_model}")

      url = LOAD_LOG_FILENAME
      df = pd.read_csv(url)
      _last_serial = df.tail(1).loc[df.index[-1], df.columns[1]]
      _elo = [df.tail(1).loc[df.index[-1], df.columns[5]] ,
              df.tail(1).loc[df.index[-1], df.columns[7]] ,
              df.tail(1).loc[df.index[-1], df.columns[9]] ,
              df.tail(1).loc[df.index[-1], df.columns[11]] ,
              #df.tail(1).loc[df.index[-1], df.columns[17]] ,
              #df.tail(1).loc[df.index[-1], df.columns[20]] 
              ]
      _elo_max = df.iloc[:, 5].max()

      print("Load log ===================================> ")
      print(f"log file : {LOAD_LOG_FILENAME}")
      print(f"elo: {_elo}")
      print(f"max elo: {_elo_max}")

      self.best_elo_0 = _elo_max
      self.training_start_epoch =int( df.tail(1).loc[df.index[-1], df.columns[1]])
      self.algo_pool[0].elo_rate=_elo[0]
      self.algo_pool[1].elo_rate=_elo[1]
      self.algo_pool[2].elo_rate=_elo[2]
      self.algo_pool[3].elo_rate=_elo[3]
      #self.algo_pool[4].elo_rate=_elo[4]
      #self.algo_pool[5].elo_rate=_elo[5]

      print(f"!!! Load model {init_model}, best elo {self.best_elo_0}, start epoch {self.training_start_epoch }")

    pass
    
    is_stored_model = False
    while not is_stored_model:
      open(f'current_policy.model.lock', 'w').close()

      _file_current_policy = f'current_policy.model'
      self.algo_pool[0].link_save_model(_file_current_policy )

      _file_best_policy = f'best_policy.model'
      self.qkv_best_player.link_save_model(_file_best_policy )

      if SQL_TYPE==2:
        store_file_in_mariadb(_file_current_policy, db_config)
      pass
      try:
        os.remove(f'current_policy.model.lock')
        is_stored_model=False
        break
      except Exception as e:
        print(f"An unexpected error occurred: {e} os.remove(f'current_policy.model.lock')")
        is_stored_model=False
        break
      pass
      print("wait for model")
      time.sleep(1.0)
    pass

    # 用來後面亂數配對互打用
    # self.algo_index_list=[0,1,2,3,4,5]
    # random.shuffle(self.algo_index_list)
    # => self.algo_index_list: [3,1,5,2,4,0]
    self.algo_index_list = []
    _num_of_algo = len(self.algo_pool)
    if(_num_of_algo%2 != 0):
      raise Exception("_num_of_algo should be even")
    pass
    for i in range(len(self.algo_pool)):
      self.algo_index_list.append(i)
    pass
    
  pass

pass

# re-factor done
def Train_expand_samples(self, play_data):
  """augment the data set by rotation and flipping
  play_data: [(state, mcts_prob, winner_z), ..., ...]
  """
  expand_data = []
  for state, mcts_porb, winner in play_data:
    for i in [1, 2, 3, 4]:
      # 90度翻轉
      expanded_equal_state = np.array([np.rot90(s, i) for s in state])
      expanded_equal_mcts_prob = np.rot90(np.flipud(
          mcts_porb.reshape(BOARD_HEIGHT, BOARD_WIDTH)), i)
      
      # 將90度翻轉的資料，存到陣列裡
      expand_data.append((expanded_equal_state,
                          np.flipud(expanded_equal_mcts_prob).flatten(),
                          winner))
      # 針對這個90度翻轉，再一次水平翻轉
      expanded_equal_state = np.array([np.fliplr(s) for s in expanded_equal_state])
      expanded_equal_mcts_prob = np.fliplr(expanded_equal_mcts_prob)

      # 將水平翻轉的資料，存到陣列裡
      expand_data.append((expanded_equal_state,
                          np.flipud(expanded_equal_mcts_prob).flatten(),
                          winner))
    pass
  pass
  return expand_data
pass
TrainProcedure.expand_samples=Train_expand_samples

# re-factor done
def Train_run_self_play(self, ai_player, in_game, is_shown=0, temp=1e-3):
  """ start a self-play game using a MCTS player, reuse the search tree,
  and store the self-play data: (state, mcts_probs, z) for training
  """
  in_game.board.init_board()
  p1, p2 = in_game.board.players
  states, mcts_probs, current_players = [], [], []

  end    = None
  winner = None
  while True:
    # 請玩家下一個決策(move)，以及決策依據(move_probs)
    # input: 
    #  1. 盤面: in_game.board
    #  2. 決策噪音(模擬現實生活): temp
    #  3. 回傳的依據(機率)總和必須是1: return_prob=1
    # output:
    #  1. 決策(move)
    #  2. 決策依據(move_probs)，即盤面上每一點的權重，我們可以把這個權重當作勝率
    move, move_probs = ai_player.get_action(in_game.board,
                                         temp=temp,
                                         is_self_play=True)
    
    # store the data: 
    # 1.盤面(in_game.board.current_state())
    # 2.決策依據(move_probs)
    # 3.目前玩家是誰(1: 1號玩家，2: 2號玩家)
    states.append(in_game.board.current_state())
    mcts_probs.append(move_probs)
    current_players.append(in_game.board.current_player)

    # perform a move
    in_game.board.do_move(move)
    if is_shown:
      in_game.graphic(in_game.board, p1, p2)
    pass
    
    # 回傳目前遊戲是否終止，如果終止，贏家是誰
    end, winner = in_game.board.game_end()

    # 遊戲終止的話，離開迴圈
    if end:
      break
    pass # if end
  pass # while end

  # winner from the perspective of the current player of each state
  winners_z = np.zeros(len(current_players))
  if winner != -1:
    winners_z[np.array(current_players) == winner] = 1.0
    winners_z[np.array(current_players) != winner] = -1.0
  pass

  # reset alphazero decision tree
  ai_player.link_reset()

  if is_shown:
    if winner != -1:
        print("Game end. Winner is player:", winner)
    else:
        print("Game end. Tie")
    pass
  pass

  # 將set轉成list格式
  # zip([1,2,3], [11,22,33], [111,222,333]]
  # => [(1,11,111),(2,22,222),(3,33,333)]

  # list([(1,11,111),(2,22,222),(3,33,333)])[:]
  # => [[1,11,111],[2,22,222],[3,33,333]]
  play_record = list(zip(states, mcts_probs, winners_z))[:]

  return winner, play_record
pass
TrainProcedure.run_self_play=Train_run_self_play

# re-factor done 修改完成
def Train_policy_evaluate(self, n_games=10):
  """
  Evaluate the trained policy by playing against the pure MCTS player
  Note: this is only for monitoring the progress of training
  """

  random.shuffle(self.algo_index_list)

  _samples=copy.deepcopy(self.algo_index_list)
  i = 0
  while len(_samples)>0:
    _a = _samples.pop()
    _b = _samples.pop()
    winner = self.game.start_play(  self.algo_pool[self.algo_index_list[_a]],
                                    self.algo_pool[self.algo_index_list[_b]],
                                    start_player=i % 2,
                                    is_shown=0)
    rate = EloRating(self.algo_pool[self.algo_index_list[_a]].elo_rate, self.algo_pool[self.algo_index_list[_b]].elo_rate, winner)

    self.algo_pool[self.algo_index_list[_a]].elo_rate = rate[0]
    self.algo_pool[self.algo_index_list[_b]].elo_rate = rate[1]
    i =+ 1
  pass
  
  _log = f"# time, {show_time_now()},"

  for i in range(len(self.algo_pool)):
    _log = f"{_log} {self.algo_pool[i].log_str}, {self.algo_pool[i].elo_rate}, "
  pass

  return _log
pass
TrainProcedure.policy_evaluate=Train_policy_evaluate

# re-factor done
def Train_run(self):
  global IS_START_TRAINING

  self.algo_pool[0].load_db()

  bar = tqdm(range(GAME_BATCH_NUM), file=sys.stdout)
  for i in bar: # start training
    ####################
    # step 1. 自我對弈 PLAY_BATCH_SIZE 次
    for _ in range(PLAY_BATCH_SIZE):
      # 取得一局自我對弈紀錄：包含 贏家(winner) 與 棋譜(play_record)
      winner, play_record = self.run_self_play(  self.algo_pool[0], self.game, 
                                                  temp=TEMPERATURE)
      # 目前這一局遊戲樣本總共走了幾步
      self.episode_len = len(play_record)

      # 將現在的遊戲樣本，水平翻轉，90度翻轉，增加樣本數
      expand_play_record = self.expand_samples(play_record)

      # 將自我訓練的資料放到queue裡
      pickled_data = [(pickle.dumps(data)) for data in expand_play_record] 

      # 將自我訓練的資料放到queue裡
      insert_training_data(pickled_data)
    pass
    
    _msg = f"games: {i}, episode_len:{self.episode_len}"
    print( f" => {_msg}",end='')
    
    self.algo_pool[0].load_db()

    ####################
    # step 2. 開始訓練
    if self.algo_pool[0].data_buffer_size() > BATCH_SIZE:

      if not IS_START_TRAINING: 
        print(f"\n\n####  start training  (data set size: {self.algo_pool[0].data_buffer_size()}) ##### \n\n")
        IS_START_TRAINING=True
      pass

      loss, entropy, _log = self.algo_pool[0].train_policy()

      log_file_path = f"{self.algo_pool[0].__class__.__name__}_{LOG_FILE_TRAINING}"

      try:
        with open(log_file_path, "a") as log_file:
            print(_log, file=log_file)
      except Exception as e:
          print(f"An unexpected error occurred: {e}")
          time.sleep(1.2)
          with open(log_file_path, "a") as log_file:
            print(_log, file=log_file)
      pass
    pass

    ####################
    # step 3. 存檔，並檢查目前模型的能力，
    if (i) % CHECK_FREQ == 0:
      
      # Step 0. 讀新的db進來(與評估模型無關)
      self.algo_pool[0].load_db()

      # Step 3.1. 先存檔
      # self.alpha_player.link_save_model( f'{TRAINED_MODELS_DIR}/current_policy_at_{i}.model')
      self.algo_pool[0].link_save_model( f'{TRAINED_MODELS_DIR}/{self.algo_pool[0].__class__.__name__}_current_policy.model')

      open(f'current_policy.model.lock', 'w').close()

      _file_current_policy = f'current_policy.model'
      self.algo_pool[0].link_save_model(_file_current_policy )

      try:
        with open("log_save_model.csv", "a") as log_file:
            _log=f"{i}, {show_time_now()}"
            print(_log, file=log_file)
      except Exception as e:
          print(f"An unexpected error occurred: {e}")
          time.sleep(1.2)
          with open("log_save_model.csv", "a") as log_file:
            _log=f"{i}, {show_time_now()}"
            print(_log, file=log_file)
      pass

      try:
        os.remove(f'current_policy.model.lock')
      except Exception as e:
          print(f"An unexpected error occurred: {e} os.remove(f'current_policy.model.lock')")
          
      pass

      if IS_ENABLE_SELF_PLAY:
        for z in range(NUM_OF_SELF_PALY):
          # 取得一局自我對弈紀錄：包含 贏家(winner) 與 棋譜(play_record)
          winner, play_record = self.run_self_play(  self.qkv_best_player, self.game, 
                                                      temp=TEMPERATURE)
          # 目前這一局遊戲樣本總共走了幾步
          self.episode_len = len(play_record)

          # 將現在的遊戲樣本，水平翻轉，90度翻轉，增加樣本數
          expand_play_record = self.expand_samples(play_record)

          # 將自我訓練的資料放到queue裡
          pickled_data = [(pickle.dumps(data)) for data in expand_play_record] 

          # 將自我訓練的資料放到queue裡
          insert_training_data(pickled_data)
          print(f"\n# self_play_game_thread_work: add training samples : now samples {self.algo_pool[0].data_buffer_size()} ", flush=True)
        pass
      pass

      # Step 3.2. 評估目前模型能力
      
      if  IS_ENABLE_SELF_EVALUATE:
        # Step 3.2.1. 計算勝率
        _log = self.policy_evaluate()

        _log = f"game_index,{i},{_log}"

        _ai_elo = self.algo_pool[0].elo_rate

        if (_ai_elo > self.best_elo_0):
          self.best_elo_0 = _ai_elo
          print(f"new record {self.best_elo_0 } at {i}")
          self.algo_pool[0].link_save_model( f'{TRAINED_MODELS_DIR}/best_elo_{self.algo_pool[0].__class__.__name__}_{self.best_elo_0 }.model')        
          self.algo_pool[0].link_save_model( f'best_policy.model')        
          self.qkv_best_player.link_load_model(f'best_policy.model')

          if SQL_TYPE==2:
            store_file_in_mariadb(f'best_policy.model', db_config)
            print(f"#######!!! upload new record {self.best_elo_0 } at {i} to db ")
          pass
        pass

        print("\n ## EVALUATION: current self-play game index: {}, sample size: {}".format(i,self.algo_pool[0].data_buffer_size()))
        print(f"\n{_log}")

        log_file_path = f"{self.algo_pool[0].__class__.__name__}_{LOG_FILE_EVALUATION}"
        readable = os.access(log_file_path, os.R_OK)
        writable = os.access(log_file_path, os.W_OK)
        executable = os.access(log_file_path, os.X_OK)

        try:
          with open(log_file_path, "a") as log_file:
              print(_log, file=log_file)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(1.2)
            with open(log_file_path, "a") as log_file:
              print(_log, file=log_file)
        pass
      pass

    pass # end of step 3.

  pass # end training
pass
TrainProcedure.run=Train_run


if __name__ == '__main__':
  train_process = TrainProcedure()
  train_process.run()
pass