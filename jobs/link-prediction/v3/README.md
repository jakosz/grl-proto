This is very similar to v2, changes include: 
* drop noise-contrastive estimation
* fix number of nodes to 50
* for each run sample a different graph (using the same parameters)

Example usage:
```bash
$ python3 run.py --config config.yaml --output results.json
```

I takes about 3 minutes on a 96-core machine to complete one run. 
