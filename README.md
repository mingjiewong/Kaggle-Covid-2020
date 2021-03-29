# Kaggle Covid-19 mRNA Vaccine Degradation Prediction 2020

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The repository contains a [Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))-based solution to the [OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction Competition](https://www.kaggle.com/c/stanford-covid-vaccine) held on Kaggle between September 11 and October 7, 2020. Check out my [profile](https://www.kaggle.com/mwong007)!

![image](https://github.com/mingjiewong/Kaggle-Covid-2020/blob/master/Figure1.png)

## Getting Started

Clone the repo:
```
git clone https://github.com/mingjiewong/Kaggle-Covid-2020.git
cd Kaggle-Covid-2020
```

Download raw data from Kaggle at ```https://www.kaggle.com/c/stanford-covid-vaccine/data``` and extract it:
```
mkdir {path-to-dir}/Kaggle-Covid-2020/datasets
cd {path-to-dir}/Kaggle-Covid-2020/datasets
unzip stanford-covid-vaccine.zip
```

Install dependencies using **python 3.8**:
```
pip install -r requirements.txt
```

Run the model (from root of the repo):
```
python main.py
```

## Acknowledgements
***
* Special thanks to my other team member: [Jing](https://www.kaggle.com/jinghuiwong)
* Solution inspired by both [CPMP](https://www.kaggle.com/cpmpml/graph-transfomer) and [mrkmakr](https://www.kaggle.com/mrkmakr/covid-ae-pretrain-gnn-attn-cnn)
