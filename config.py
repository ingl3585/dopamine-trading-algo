# config.py

class Config:
    FEATURE_FILE = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features\\features.csv"
    MODEL_PATH   = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\model\\rl\\actor_critic_model.pth"

    INPUT_DIM   = 4
    HIDDEN_DIM  = 64  # Reduced from 128 - simpler model
    ACTION_DIM  = 3
    LOOKBACK    = 1   # Removed LOOKBACK since we're not using sequences

    BATCH_SIZE  = 32  # Reduced for more frequent training
    GAMMA       = 0.95  # Slightly reduced discount factor for trading
    ENTROPY_COEF= 0.01  # Reduced entropy coefficient
    LR          = 1e-4  # Reduced learning rate for stability

    BASE_SIZE   = 4
    MAX_SIZE    = 10
    MIN_SIZE    = 1

    TEMPERATURE = 0.8  # Slightly higher for more exploration