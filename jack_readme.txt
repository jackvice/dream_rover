python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs atari \
  --run.train_ratio 32
  

python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs bipedalwalker



python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs dmc_proprio
