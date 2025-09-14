## Alleviating Overconfidence in Long-Tailed Time Series Classification with Hierarchical  Reverse Distillation

### Datasets
We take the HAR dataset as an example and construct training sets with imbalance ratios of 50 and 100, respectively.

### Environment
- Python 3.11
- Pytorch 2.4


### Run
`$ python main_run.py --loss_type mse`
`$ python main_run.py --loss_type kl_js`
