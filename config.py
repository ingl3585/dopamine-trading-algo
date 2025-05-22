# config.py

class Config:
    FEATURE_FILE = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features\\features.csv"
    MODEL_PATH   = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\model\\rl\\actor_critic_model.pth"

    INPUT_DIM   = 4
    HIDDEN_DIM  = 64
    ACTION_DIM  = 3
    LOOKBACK    = 1

    BATCH_SIZE  = 32
    GAMMA       = 0.95
    ENTROPY_COEF= 0.05
    LR          = 1e-4

    BASE_SIZE   = 4
    MAX_SIZE    = 10
    MIN_SIZE    = 1

    TEMPERATURE = 2.0