BATCH_SIZE = 128
DIM_RANGE = (2, 4)


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

TRAINING_STEPS = 16

VCOUNT_FROM = 7
VCOUNT_TO = 13
