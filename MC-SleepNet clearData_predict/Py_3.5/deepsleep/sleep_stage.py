import numpy as np
# Label values
W = 0
NR = 1
R = 2

NUM_CLASSES = 3  # exclude UNKNOWN

class_dict = {
    0: "W",
    1: "NR",
    2: "R",
}

EPOCH_SEC_LEN = 20  # seconds
SAMPLING_RATE = 250

#MOUSES = ["eeg_0001.mat", "eeg_0002.mat", "eeg_0003.mat"]
MOUSES = []
for i in np.arange(1,15):
    MOUSES.append("eeg_%4d.mat"% i)

def print_n_samples_each_class(labels):
    #import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print ("{}: {}").format(class_dict[c], n_samples)
