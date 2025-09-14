## Alleviating Overconfidence in Long-Tailed Time Series Classification with Hierarchical  Reverse Distillation

### Datasets
Due to the 50MB memory limit, the datasets will be made publicly available once the review process is complete.

### Baseline Models
Our benchmark includes 14 time series baseline models, which are stored in the 'Libs' folder.


### Environment
- Python 3.8
- Pytorch 2.1


### Run
`$ python main_run.py --loss_type mse`
`$ python main_run.py --loss_type kl_js`
