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

    # CONSERVATIVE: Smaller position sizes with better position awareness
    BASE_SIZE   = 6     # Reduced from 15 to 6 (max single trade)
    MAX_SIZE    = 10    # This is your total position limit
    MIN_SIZE    = 1     # Minimum trade size

    TEMPERATURE = 2.0