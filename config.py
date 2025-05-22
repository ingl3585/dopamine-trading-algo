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

    # FIXED: Increase BASE_SIZE for more size variation
    BASE_SIZE   = 10    # INCREASED from 6 to 10
    MAX_SIZE    = 15    # INCREASED proportionally 
    MIN_SIZE    = 1     

    TEMPERATURE = 2.0