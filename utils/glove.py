from utils import constants

def train(cooccurences, d=50, alpha=0.75, x_max=100.0, epochs=25, batch_size=10, workers=1):
    # glv = {"token": []}
    # # TODO: implement
    # glv.save = save
    # return glv
    model = glove.Glove(cooccurences, d=d, alpha=alpha, x_max=x_max)
    for epoch in range(epochs):
        err = model.train(batch_size=batch_size, workers=workers, verbose=True)
        print("epoch %d, error %.3f" % (epoch, err), flush=True)
    return model.W


def save(glv, filename=constants.GLOVE_FILE):
    import numpy as np
    with open(filename, "wb") as file:
        np.savez(file, glv)