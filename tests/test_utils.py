from grl.utils import *


def test_atog_gtoa():
    n = np.random.randint(2, 1024)
    p = np.random.uniform(.01, .99)    
    A = get_adj(n, p)
    assert np.all(gtoa(atog(A)) == A)
    

def test_anti_edgelist():
    print("test_anti_edgelist: raise NotImplementedError")

    
def test_edgelist():
    n = 1024
    p = .1
    A = get_adj(n, p)
    G = atog(A)
    assert edgelist(A).shape[0] == G.ecount()
    

def test_get_adj_density():
    n = 1024
    p = .1
    A = get_adj(n, p)
    G = atog(A)
    assert squareform(A).mean() == G.density()
    assert A.mean() > 0.095 and A.mean() < 0.105
    assert G.density() > 0.095 and G.density() < 0.105
    

def test_get_adj_shape():
    n = np.random.randint(2, 1024)
    p = 0.5
    assert get_adj(n, p).shape == (n, n)
    

def test_get_adj_symmetry():
    n = np.random.randint(2, 1024)
    p = np.random.uniform(.01, .99)
    A = get_adj(n, p)
    assert np.all(A == A.T)
    

def test_sigmoid():
    assert sigmoid(0.) == .5
    assert sigmoid(+1e+3) == 1
    assert np.allclose([sigmoid(-1e+2)], [0])
    

def test_vcount():
    n = np.random.randint(2, 1024)
    p = np.random.uniform(.01, .99)    
    A = get_adj(n, p)
    G = atog(A)
    assert vcount(A) == vcount(G) == n
