
class Algo_Human(object):
  """
  human player
  """

  def __init__(self):
    self.player = None
  pass

  def __str__(self):
    return "Human {}".format(self.player)
  pass

pass
# 設定先手或後手
def Algo_Human_set_player_ind(self, p):
  self.player = p
pass
# 將Algo_Human_set_player_ind加入Algo_Human class裡
Algo_Human.set_player_ind=Algo_Human_set_player_ind

# 接收玩家所輸入的下子座標並處理
def Algo_Human_get_action(self, board):
  
  try:
    #利用input接收人類玩家所輸入的座標
    location = input("Your move y,x : ")
    # 確認輸入的是否為字串
    if isinstance(location, str):  # for python3
      # 以,為分割字串
      location = [int(n, 10) for n in location.split(",")]
    pass
    # 將x,y座標轉換為數字座標
    move = board.location_to_move(location)
  # 例外狀況下將move設為-1
  except Exception as e:
    move = -1
  pass
  # 若move為-1或是所選的下子座標不可使用的狀況下顯示提示字樣
  if move == -1 or move not in board.availables:
      print("invalid move")
      # 重新執行function
      move = self.get_action(board)
  pass

  # 回傳使用這個move的原因
  _log_of_move_probs = None
  # 回傳下子座標
  return move, _log_of_move_probs

pass
# 將Algo_Human_get_actionfunction加入Algo_Human class裡
Algo_Human.get_action=Algo_Human_get_action

