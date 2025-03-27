# Train with rendering
python train.py --render --timesteps 2000000

# Resume training
python train.py --resume --model_path my_saved_model

# Play trained model
python train.py --play --model_path kungfu_ppo

# Debug mode
python train.py --play --debug