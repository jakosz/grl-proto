This is very similar to v2, changes include: 
* noise-contrastive estimation is not used at all (i.e. only negative sampling)
* number of nodes is fixed to 50
* a different graph is sampled for each run of the model (from parameters specified in the config)

Example usage:
```bash
$ python3 run.py --config config.yaml --output results.json
```

I takes about 3 minutes on a 96-core machine to complete one run. 
