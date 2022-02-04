#### Graph Representation Learning

This library is an ugly, unstable and experimental bunch of hacks I write for my PhD. Don't use it. 
I publish it only to make it easier for some people to reproduce my Jupyter notebooks.  

---

#### DOCUMENTATION

To build the documentation do: 
```bash
$ cd docs && make html
```

---

#### MISCELLANEOUS
* Functions defined in `graph.core` are exported at the module level. 
* I try to mark places where conversions between 1- and 0- indexing are taking place with a `@indexing` comment. 
* Convention: functions that modify their arguments in place (without a memcpy) do not return anything. 
* Many tests are carried out against `igraph`, `numpy` and `tensorflow`. This is not very elegant, and shouldn't be *the only* case, but for initial development should suffice. 
