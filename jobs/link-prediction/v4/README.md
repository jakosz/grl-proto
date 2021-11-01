## Experiment info. 

Repository version: 0.5.2 (to be sure: `git describe --tag` gives exactly `0.5.2`, no commits ahead of tag).   
Two Hetzner CPX51 servers were used for each experiment.  

### --vcount 50: 
```bash
time seq 1 16 | parallel --ungroup python3 run.py --runs 100 --dims 21 --vcount 50 --iter $((2**21)) --st --output link-prediction-v4-50.json >> error.log
```

Times:
* 839m8.469s
* 821m29.802s

### 100-node: 
```bash
time seq 1 16 | parallel --ungroup python3 run.py --runs 100 --dims 21 --vcount 100 --iter $((2**23)) --st --output link-prediction-v4-100.json >> error.log
```

Times:
* 2518m59.522s
* 2609m30.199s
