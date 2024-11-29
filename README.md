
<!-- PROJECT LOGO -->
<br />
  <h3 align="center">DGT: Unbiased Sequential Recommendation via Disentangled Graph Transformer </h3>

<!-- TABLE OF CONTENTS -->
<!--<details open="open">-->
<!--  <summary>Table of Contents</summary>-->
<!--  <ol>-->
<!--    <li>-->
<!--      <a href="#about-the-project">About The Project</a>-->
<!--    </li>-->
<!--    <li>-->
<!--      <a href="#project-structure">Project Structure</a>-->
<!--    </li>-->
<!--    <li><a href="#usage">Usage</a></li>-->
<!--    <li><a href="#model-training-and-evaluation">Model Training and Evaluation</a></li>-->
<!--    <li><a href="#contact">Contact</a></li>-->
<!--  </ol>-->
<!--</details>-->



<!-- ABOUT THE PROJECT -->
## About The Project
This repo focuses on the debias problem of explicit feedback-based sequential recommendation. 

<!-- Project Structure -->
## Project Structure
```bash 
├── README.md                                 Read me file 
├── Dataset                                   dataset processing code
├── Datasets                                  datasets that have been processed
├── all_models                                models (our model and most of the baselines)
├── some_baselines                            DCRec, HTSR, SDHID
├── argparser.py                              argument definition
├── main_pro.py                               main script for the proposed method
├── data_loader.py                            data loader
├── data_loader_seq.py                        data loader for sequential recommendation
├── main_single.py                            baseline methods 
├── train_pro.py                              training procedure of the proposed method 
├── utils.py                                  utils such as early stop function
└── .gitignore                                gitignore file
```

<!-- USAGE -->
## Usage
To perfrom our proposed  method, please run the following command (find details about the arguments in argparser.py) :
```commandline
python main_pro.py --method pro --model GNN_rate --prefix GNN_rate_128_V2 --dataset "cd" --epoch 100 --bs 1024 --train train --latent_dim 128 --lr 0.001 --wd 0 --bar False --n_channels 2 --n_layers 1 --version 2
```

## Raw Datasets
Amazon datasets: CD, Movie, Sport, Music, Beauty (https://jmcauley.ucsd.edu/data/amazon/)
