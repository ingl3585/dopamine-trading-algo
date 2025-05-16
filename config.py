# config.py

class Config:
    FEATURE_FILE = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\features\\features.csv"
    MODEL_PATH   = r"C:\\Users\\ingle\\OneDrive\\Desktop\\Actor_Critic_ML_NT\\rl\\actor_critic_model.pth"

    INPUT_DIM   = 3
    HIDDEN_DIM  = 128
    ACTION_DIM  = 3
    LOOKBACK    = 20

    BATCH_SIZE  = 64
    GAMMA       = 0.99
    ENTROPY_COEF= 0.01
    LR          = 5e-4

    BASE_SIZE   = 5
    CONS_SIZE   = 2
    MIN_SIZE    = 1

    TEMPERATURE = 0.75