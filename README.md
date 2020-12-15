# SCAN

Unofficial Pytorch implementation of SCAN: Learning to Classify Images without Labels (ECCV 2020) [[paper](https://arxiv.org/abs/2005.12320)]

## How to run

Install requirements

```bash
pip3 install -r requirements.txt
```

Training

```bash
# simclr
python3 main.py -c configs/simclr.yaml -o <path-to-output> (default: ./dataset/simclr/run-date-time)

# clustering
python3 main.py -c configs/cluster.yaml -o <path-to-output> (default: ./dataset/cluster/run-date-time)

# self label
python3 main.py -c configs/selflabel.yaml -o <path-to-output> (default: ./dataset/selflabel/run-date-time)
```

