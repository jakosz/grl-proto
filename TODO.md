Scaling:
- [ ] Evaluation on graphs too large to build adjacency matrix.
- [ ] Online evaluation + checkpointing for training larger graphs. 

Embedding methods:
- [ ] Orthogonalization of simple embeddings. 
- [ ] Global link prediction heuristics. 

Functionality:
- [ ] add/remove edges

Performance:
- [ ] `graph.utils.to_igraph` is too slow.  
- [ ] `jobs/performance-tuning`:
    - [ ] learning rate
    - [ ] describe tensorflow benchmarks

#### TODO
- [ ] add full matrix reconstruction tests for symmetric and diagonal models.  
- [ ] diagonal model is 2x slower than symmetric and asymmetric. it shouldn't be. investigate.  
- [ ] evaluation for larger graphs should not rely on full adjacency matrix. 
- [ ] wrap encode-decode-evaluate in one function. do I want to wrap embeddings in an object?
- [ ] AUC and accuracy computed in numba-jitted functions.
- [ ] port graph to shmem.
- [ ] loss/metrics monitoring during training. Automatic stopping.
- [ ] Adam optimizer. 

#### DOCUMENTATION 
- [ ] Sphinx, docstrings & RTD. 
- [ ] List reasons to use 1-indexing: padding, missing data. 
- [ ] grl vs tf benchmarks.
- [ ] learning rate & number of steps conditional on graph size/density.

#### OTHER
- [ ] Global link prediction heuristics. 
- [ ] Built-in real-world datasets.  

