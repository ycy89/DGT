# Parameter tuning on the movie dataset
# wd = 0
./run.sh main_single.py --method trad --model MF_rate --prefix MF_rate_16_wd0 --dataset movie --epoch 100 --bs 1024 --train train --latent_dim 16 --lr 0.001 --wd 0

./run.sh main_single.py --method trad --model MF_rate --prefix MF_rate_32_wd0 --dataset movie --epoch 100 --bs 1024 --train train --latent_dim 32 --lr 0.001 --wd 0

./run.sh main_single.py --method trad --model MF_rate --prefix MF_rate_64_wd0 --dataset movie --epoch 100 --bs 1024 --train train --latent_dim 64 --lr 0.001 --wd 0

./run.sh main_single.py --method trad --model MF_rate --prefix MF_rate_128_wd0 --dataset movie --epoch 100 --bs 1024 --train train --latent_dim 128 --lr 0.001 --wd 0

#  l2 wd = 1e-5
./run.sh main_single.py --method trad --model MF_rate --prefix MF_rate_16_wd1-5 --dataset movie --epoch 100 --bs 1024 --train train --latent_dim 16 --lr 0.001 --wd 0.00001

./run.sh main_single.py --method trad --model MF_rate --prefix MF_rate_32_wd1-5 --dataset movie --epoch 100 --bs 1024 --train train --latent_dim 32 --lr 0.001 --wd 0.00001

./run.sh main_single.py --method trad --model MF_rate --prefix MF_rate_64_wd1-5 --dataset movie --epoch 100 --bs 1024 --train train --latent_dim 64 --lr 0.001 --wd 0.00001

./run.sh main_single.py --method trad --model MF_rate --prefix MF_rate_128_wd1-5 --dataset movie --epoch 100 --bs 1024 --train train --latent_dim 128 --lr 0.001 --wd 0.00001
