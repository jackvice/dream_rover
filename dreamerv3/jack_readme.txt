python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs atari \
  --run.train_ratio 32
  

python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs bipedalwalker

don't forget the enc and dec in configs.yaml, you dumb ass.

