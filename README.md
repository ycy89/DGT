<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT LOGO -->
<br />
  <h3 align="center">Debias Cross-Domain Sequential Recommendation</h3>

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
├── Dataset                                   dataset processing
├── all_models                                models 
├── argparser.py                              arguement definition
├── main_pro.py                               main script for the proposed method
├── data_loader.py                            data loader
├── data_loader_seq.py                        data loader for sequential recommendation
├── main_single.py                            baseline methods 
├── train_pro.py                              training procedure of the proposed method 
├── utils.py                                  utils such as early stop function
├── README.md                                 readme file
└── .gitignore                                gitignore file
```

<!-- USAGE -->
## Usage
```commandline
python main_single.py --method trad --model MF_rate --prefix MF_rate_64 --dataset cd --epoch 100 --bs 1024 --train train --latent_dim 64    

python main_pro.py --method pro --model GNN_rate --prefix GNN_rate_128_V2 --dataset "cd" --epoch 100 --bs 1024 --train train --latent_dim 128 --lr 0.001 --wd 0 --bar False --n_channels 2 --n_layers 1 --version 2 --main_path "/cfs/cfs-a769xugn/chenglin/DHIN_Debias"
```
Find details about the arguments in argparser.py 

## Dataset 
Amazon datasets: CD, Movie, Sport, Music, Beauty (https://jmcauley.ucsd.edu/data/amazon/)
