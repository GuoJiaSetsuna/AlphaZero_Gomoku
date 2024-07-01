#################################
# for play
# PLAY_BOARD_WIDTH = 6
# PLAY_BOARD_HEIGHT = 6
# PLAY_N_IN_ROW = 4
# PLAY_MODEL_PATH = "trained_models/6x6x4/best_policy_6x6x4_mcts_10000.model"
PLAY_BOARD_WIDTH = 8
PLAY_BOARD_HEIGHT = 8
PLAY_N_IN_ROW = 5
PLAY_MODEL_PATH = "trained_models\8x8x5/best_policy_8x8x5_mcts_10000.model"

PLAY_C_PUCT = 5
PLAY_MCTS_SIM_TIMES = 30000
PLAY_ALPHAZERO_SIM_TIMES = 400

PLAY_START_PLAYER = 1
PLAY_IS_SHOWN = 1

IS_ENABLE_SELF_EVALUATE = True
NUM_OF_SELF_PALY = 1
IS_ENABLE_SELF_PLAY = True

#################################
# for training 
# params of the board and the game
BOARD_WIDTH = 8
BOARD_HEIGHT = 8
N_IN_ROW = 5
# BOARD_WIDTH = 6
# BOARD_HEIGHT = 6
# N_IN_ROW = 4

# training params
LEARN_RATE = 2e-3
# adaptively adjust the learning rate based on KL
LR_MULTIPLIER = 1.0  
LR_MULTIPLIER_LOWER_BOUND = 0.1
LR_MULTIPLIER_UPPER_BOUND = 10.0
LR_MODIFY_RATE = 1.5

# the temperature param
TEMPERATURE = 1.0  
# num of simulations for each move
N_ALPHAZERO_SIMULATE = 500  
C_PUCT = 5

# mini-batch size for training
BATCH_SIZE = 512  
MIN_TRAIN_DATA_SIZE = 2000
BUFFER_SIZE = 10000

# 玩多少次遊戲，訓練一次
PLAY_BATCH_SIZE = 1 

# num of train_steps for each update
EPOCHS = 5  

KL_TARG = 0.02
KL_TARG_UPPER_BOUND = KL_TARG * 2.0
KL_TARG_LOWER_BOUND = KL_TARG / 2.0

MAX_KL_IN_ONE_LEARNING = KL_TARG * 4.0


# 每50場評估一次alphazero訓練成果
CHECK_FREQ = 25 #50 

# 總共要訓練幾場
GAME_BATCH_NUM = 10000

# 要讀取的model file
LOAD_MODEL_FILENAME = "trained_models/AlphaZero_Player_current_policy.model"
LOAD_BEST_MODEL_FILENAME = "best_policy.model"

# 要讀取的previous log file
LOAD_LOG_FILENAME = "AlphaZero_Player_z_log_evaluation.csv"

# 可以更新模型的勝率門檻
MODEL_UPDATE_WIN_RATIO = 0.9
# 可以更新模型，一次更新的難度
MODEL_UPDATE_SCALE = 500
# 模型的目標難度
MODEL_UPDATE_MAX_SCALE = 10000

# num of simulations used for the pure mcts, which is used as
# the opponent to evaluate the trained policy
DEFAULT_PURE_MCTS_PLAYOUT_NUM = 500

LOG_FILE_EVALUATION="z_log_evaluation.csv"
LOG_FILE_TRAINING="z_log_training.csv"


IS_START_TRAINING = False

TRAINED_MODELS_DIR = 'trained_models'

# sql type
# 1: sqlite
# 2: mariadb
SQL_TYPE = 1

db_in_memory = ":memory:"
db_in_file = "sampleDB/pickled_objects.db"
db_out_file = "pickled_objects_out.db"
table_name = "pickled_objects"
MARIA_DB_MODULE_TABLE = "files_2"

db_config = {
    "host": "...",
    "user": "...",
    "password": "1234",
    "database": "32132",
    "port": 1234
}