# Sequence Prediction

## TO-DO
### Datset
- - [x] conll2003
- - [ ] atis
### Neural NER
- - [ ] CharLSTM+WordLSTM+CRF: [Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf)
  - - [x] Make a CoNLL-2003 batcher
  - - [x] Implement trainer
  - - [x] Implement WordLSTM + softmax
  - - [x] Implement CharLSTM + WordLSTM + softmax
  - - [ ] Implement WordLSTM + CRF
  - - [ ] Implement CharLSTM + WordLSTM + CRF

### Slot Filling + intent prediciton
- - [ ] [Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling](https://arxiv.org/abs/1609.01454)

### Tree VAE
- - [ ] [STRUCTVAE: Tree-structured Latent Variable Models for Semi-supervised Semantic Parsing](https://arxiv.org/abs/1806.07832)

## Requiremet (python 3)
```
conda install pytorch  -c pytorch

```
CoNLL-2003 can be downloaded from https://www.clips.uantwerpen.be/conll2003/ner/
