class Config:
    DATA_PATH = './dataset'
    OUTPUT_DIR = './checkpoints'
    
    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-5
    TRAIN_SPLIT = 0.8
    
    STEP_SIZE = 2
    GAMMA = 0.1
    
    NUM_WORKERS = 4
    
    USE_MIXED_PRECISION = True
    USE_GRADIENT_CHECKPOINTING = True
    
    MODEL_NAME = 'google/vit-base-patch16-224-in21k'