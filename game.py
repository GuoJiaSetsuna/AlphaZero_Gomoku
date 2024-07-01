from __future__ import print_function
import numpy as np
import time
import pandas as pd

# 
class Board(object):
    """board for the game"""
    # 宣告棋盤大小與勝利子數變數並且設定預設值
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2
    # 檢查勝利子數是否大於棋盤大小與宣告先手玩家、可下子位置
    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
    # 以位置數字計算出座標
    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]
    # 以座標計算出位置數字
    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move
    #翻轉盤面取得當前玩家棋盤狀態
    def current_state(self):
      """return the board state from the perspective of the current player.
      state shape: 4*width*height
      """

      square_state = np.zeros((4, self.width, self.height))
      if self.states:
        moves, players = np.array(list(zip(*self.states.items())))
        move_curr = moves[players == self.current_player]
        move_oppo = moves[players != self.current_player]
        square_state[0][move_curr // self.width,
                        move_curr % self.height] = 1.0
        square_state[1][move_oppo // self.width,
                        move_oppo % self.height] = 1.0
        # indicate the last move location
        square_state[2][self.last_move // self.width,
                        self.last_move % self.height] = 1.0
      pass

      if len(self.states) % 2 == 0:
        square_state[3][:, :] = 1.0  # indicate the colour to play
      pass

      return square_state[:, ::-1, :]
    pass
    # 將所下子的位置記錄下子玩家並從可移動的位置移除，最後切換當前玩家並替換記錄最後下子位置
    def do_move(self, move):
      self.states[move] = self.current_player
      self.availables.remove(move)
      self.current_player = (
          self.players[0] if self.current_player == self.players[1]
          else self.players[1]
      )
      self.last_move = move
    pass
    # 檢測棋盤上是否有勝利玩家
    def has_a_winner(self):
      width = self.width
      height = self.height
      states = self.states
      n = self.n_in_row

      moved = list(set(range(width * height)) - set(self.availables))
      if len(moved) < self.n_in_row *2-1:
          return False, -1

      for m in moved:
        h = m // width
        w = m % width
        player = states[m]

        if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
            return True, player
        pass

        if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
            return True, player
        pass

        if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
            return True, player
        pass

        if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
            return True, player
        pass
      pass

      return False, -1
    pass
    # 檢測遊戲是否結束
    # 首先檢測是否有玩家勝利如果為True則回傳True與勝利者，反之若為False則進到下一步
    # 第二步檢測是否已無可下子位置如果為True則回傳True與-1，反之則下一步
    # 最後檢測結束尚未結束回傳False與-1
    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""
    # 將棋盤變數匯入
    def __init__(self, board, **kwargs):
        self.board = board
    # 畫出棋盤資訊
    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        # 設定棋盤大小
        width = board.width
        height = board.height
        # 顯示各玩家在棋盤上的對應標誌
        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        # 畫出棋盤
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')
    # 啟動遊戲
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        # 確認變數內是否錯誤
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        # 設定先手玩家
        self.board.init_board(start_player)
        # 宣告玩家變數
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        data_dict = {
            'play': [],
            'runtime': []
        }
        data_dict2 = {
            'play': [],
            'runtime': []
        }
        # 畫出盤面(初始狀態)
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        # 遊戲執行
        while True:
            # 當前執子玩家變數設置
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            # start_time = time.time()
            # 取得執子玩家所下子座標(algo_human.py、algo_pure_mcts.py:get_action)
            move, _log_of_move_probs = player_in_turn.get_action(self.board)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # if self.board.get_current_player() == 1 :
            #     data_dict['play'].append(self.board.get_current_player())
            #     data_dict['runtime'].append(execution_time)
            # elif self.board.get_current_player() == 2 :
            #     data_dict2['play'].append(self.board.get_current_player())
            #     data_dict2['runtime'].append(execution_time)
            # 進行下子並切換執子玩家
            self.board.do_move(move)
            # 重新印出下子後的棋盤
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            # 宣告與偵測遊戲是否結束決定是否結束遊戲運行
            end, winner = self.board.game_end()
            # end如果為ture代表遊戲結束
            if end:
                # df = pd.DataFrame(data_dict)
                # df.to_csv('data.csv', index=False)
                # df2 = pd.DataFrame(data_dict2)
                # df2.to_csv('data2.csv', index=False)
                if is_shown:
                    # 若winner不等於-1代表有玩家勝出印出提示字樣
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    # 剩餘情況代表盤面上已無可下子座標顯示提示字樣
                    else:
                        print("Game end. Tie")
                return winner
