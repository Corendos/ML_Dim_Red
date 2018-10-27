import os
import sys

sys.path.append(os.path.join(os.getcwd(), os.pardir))

from datasets import digits, phoneme
from sklearn import decomposition, cluster, metrics, neural_network, model_selection
import matplotlib.pyplot as plt
import numpy as np

DATASET = digits
N_COMPONENTS = 11
N_CLUSTERS = 10
MODE = "report"

pca = decomposition.PCA(N_COMPONENTS)
new_data = pca.fit_transform(DATASET.training_features)

report = {}
labels = []

kmeans = cluster.KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(new_data)

for i, c in enumerate(kmeans.labels_):
    if c not in report:
        report[c] = {}
    real_label = DATASET.training_labels[i]
    if real_label not in labels:
        labels.append(real_label)
    if real_label not in report[c]:
        report[c][real_label] = 1
    else:
        report[c][real_label] += 1

if MODE == "pretty":
    for key in sorted(report):
        value = report[key]
        print("Cluster #{}".format(key))
        max_label_count = 0
        max_label = 0
        for key2 in sorted(value):
            value2 = value[key2]
            if value2 > max_label_count:
                max_label_count = value2
                max_label = key2
            print("\tLabel {}: {} instance(s)".format(key2, value2))
        print("\tMost instances for label {}\n".format(max_label))
elif MODE == "report":
    final_out = ""
    for key in sorted(report):
        value = report[key]
        out = [str(key)]
        max_instance_count = 0
        max_label = None
        for label in sorted(labels):
            if label in value.keys():
                out.append(str(value[label]))
                if value[label] > max_instance_count:
                    max_instance_count = value[label]
                    max_label = label
            else:
                out.append('0')
        out.append(str(max_label))
        out = ' & '.join(out) + "\\\\ \\hline \n"
        final_out += out

    print(final_out)

# General Options
TITLE = 'Neural Network Classifier'
LAYERS_TOPOLOGY = (30,)

# Final Report Options
LEARNING_RATE = 1e-2
TOLERANCE = 1e-4

classifier = neural_network.MLPClassifier()

plot_x = []
plot_y_testing = []
plot_y_mean = []

train_sizes = np.linspace(0.1, 0.9, 30)

train_size_abs, train_scores, test_scores = model_selection.learning_curve(classifier, new_data, DATASET.training_labels,
    cv=10, train_sizes=train_sizes)

train_losses = [1 - np.array(a).mean() for a in train_scores]
test_losses = [1 - np.array(a).mean() for a in test_scores]

plt.figure()
plt.grid()
plt.xlabel('Training Set Size')
plt.ylabel('Loss')
plt.title(TITLE)
plt.plot(train_size_abs, train_losses)
plt.plot(train_size_abs, test_losses)
plt.legend(['Training', 'Testing'])
plt.show()