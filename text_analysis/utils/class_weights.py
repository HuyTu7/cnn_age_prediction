import numpy as np
import math
from collections import Counter

# labels_dict : {ind_label: count_label}
# mu : parameter to tune


def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}


# random labels_dict
#labels_dict = {0: 5763, 1: 8123, 2: 4181, 3: 998}
labels_dict = {0: 5465, 1: 7837, 2: 3957, 3: 896}
print(create_class_weight(labels_dict))