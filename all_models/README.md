**<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING!** :D
-->

<!-- PROJECT LOGO -->
<br />
  <h3 align="center">Baselines of Debiasing Cross-domain Sequential Recommender</h3>

<!-- Project Structure -->
## Project Structure
```bash 
├── README.md                          Read me file
├── Single                             Single-domain baselines 
├── Cross                              Cross-domain baselines  
└── XXX                                xxx
```

## Single-domain Baselines 
### Normal Methods
|Num |Selected | Year-Venue | Title | Source | type|
| ----| ----  | ---- | ---   | ------ | ------ |
| [1]|  |SIGIR-21| Counterfactual Data-Augmented Sequential Recommendation | | data-aug|
| [2]|  |SIGIR-21| Causerec: Counterfactual user sequence synthesis for sequential recommendation| | data-aug|
| [3]|  |CIKM-22| A Relevant and Diverse Retrieval-enhanced Data Augmentation Framework for Sequential Recommendation| | data-aug|
| [4]|  |CIKM-22| Explanation guided contrastive learning for sequential recommendation | | CL|
| [5]|  |SIGIR-22| Resetbert4rec: A pre-training model integrating time and user historical behavior for sequential recommendation| | norm|


### Debiasing Methods
| Num |Selected | Year-Venue | Title | Source | type|
| --- | ----  | ---- | ---   | ------ | ------ |
|[6]| Base | WWW-22 | Unbiased Sequential Recommendation with Latent Confounders| | IPS| 
|[7]| Base | CIKM-22| Dually Enhanced Propensity Score Estimation in Sequential Recommendation| | IPS|
|[12] | Base | Recsys-22 | Denoising Self-Attentive Sequential Recommendation | | denoise| 

## Cross-domain Baselines
### Normal Methods
|Num|Selected | Year-Venue | Title | Source | type|
|----| ----  | ---- | ---   | ------ | ---|
|[8]| Base  | CIKM-22|Contrastive Cross-Domain Sequential Recommendation| [C2DSR](https://github.com/cjx96/C2DSR)| |
|[9]| Base  | TKDD-22|Mixed Information Flow for Cross-domain Sequential Recommendations| | |
|[10]| Base | TKDE-22|Parallel Split-Join Networks for Shared Ac- count Cross-domain Sequential Recommendations| | |


### Debiasing Methods
|Selected | Year-Venue | Title | Source | 
| ----  | ---- | ---   | ------ | 

### Graph-Based Methods
|Num |Selected | Year-Venue | Title | Source | 
| --- | ----  | ---- | ---   | ------ |
| [11] | Base | CIKM-2021 |Continuous-time sequential recommendation with temporal graph collaborative transformer|[TGSRec](https://github.com/dygrec/tgsrec)|

   
### Selected baselines 
1. Traditional baselines   
   
   BPRMF, NCF

2. Sequential baselines

   SASRec, GUR4Rec++, 

3. Graph methods 

   SR-GNN, TGSRec

4. Debias/Augmentation methods


   

## Experiment Design
The experiments are conducted to answer the following research questions: 
1. Overall performance.

    RQ1. How does our debiasing method performs comparing with the SOTA sequential recommendation algorithm？
      
         Single-domain Baseline sequential recommendation methods: SASRec, Bert4Rec, TGSRec

2. Ablation studies
    

<!-- CONTACT -->
## Contact

Chenglin Li - ch11 at ualberta.ca
