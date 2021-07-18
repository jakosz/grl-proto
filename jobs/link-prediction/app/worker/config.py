BATCH_SIZE = 128 
DIM_RANGE = (2, 16)


RGM_SAMPLING_SPACE = {
    'Barabasi': {
        'm': (1, 2)
    },
    'Erdos_Renyi': {
        'p': (1e-3, 1e-1)
    },
    'Forest_Fire': {
        'fw': (1e-2, 1.),
        'bw': 0.,
        'ambs': (1, 2)
    },
    'GRG': {
        'radius': (1e-2, .5)
    }
}

TRAINING_STEPS = 4096 

VCOUNT_FROM = 16 
VCOUNT_TO = 256 
