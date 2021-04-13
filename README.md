# Cancer Reports Mining
(in progress)

## Requirements
developed and tested with python 3.8

+ install packages: `pip install -r requirements.txt`

+ install italian language for spacy: `python -m spacy download it`

### GPU requirements (optional)
developed and tested with CUDA 10.0.130

+ install CUDA with conda: `conda install cudatoolkit=10.0 cudnn`

+ or install CUDA without conda: https://developer.nvidia.com/cuda-10.0-download-archive

## Datasets preparation
The data are not publicly available. The data was provided by ISPRO, an association that collects oncological reports from all over Tuscany.

`python3 prepare_old_dataset.py`

`python3 prepare_new_dataset.py`

## Experiments
### How to train models
`python3 train.py [args]`
#### Example: train a model on the old dataset, to create a classifier that filters only breast cancer reports
`python3 train.py -d datasets/all_cancer_types -m models.transformer.Transformer -ma '{"embedding_dim":256,"deep_features":256,"dropout":0.1,"num_heads":4,"n_layers":1}' -lr 1e-4 -tc sede_icdo3 -lp '{"sede_icdo3":[{"type":"regex_sub","subs":[["^C5.*",1],["^C.*",0]]},{"type":"filter","valid_set":[0,1]}]}' -ml 100 -e 10 -pt max`

#### Example: train a multi-task model on the new dataset
`python3 train.py -m models.transformer.Transformer -ma '{"embedding_dim":256,"deep_features":256,"dropout":0.1,"num_heads":4,"n_layers":1}' -lr 1e-4 -ml 100 -gb id_paz -f cla
ssifier -fa '{"path":"trained_models/2021-04-04_09.21.13_Transformer_4E1Y1U/model_best.pth","encoded_data_column":"filter_tokens","max_length":100}' -e 100 -pt max -pr max -tc grading modalita_N modalita_T stadio_N stadio_T tipo_T -tr cerb dimen
sioni ki67 numero_sentinella_asportati numero_sentinella_positivi recettori_estrogeni recettori_progestin`

### Experiment: ...
...