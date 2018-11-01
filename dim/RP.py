import os
import sys

sys.path.append(os.path.join(os.getcwd(), os.pardir))

from datasets import digits, phoneme
from sklearn import random_projection, cluster, metrics, neural_network, model_selection
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def vector_distance(x, y):
    d = [y[i] - x[i] for i in range(len(x))]
    s = 0
    for i in range(len(d)):
        s += d[i]*d[i]
    return sqrt(s)

def image_from_features(features):
    image = [[features[y * 8 + x] for x in range(7)] for y in range(8)]
    return image

DATASET = phoneme
N_COMPONENTS = 4
N_CLUSTERS = 2
MODE = "nothing"
EXPERIMENT = 'learning'

# General Options
TITLE = 'Neural Network Classifier'
N_REPEAT = 2

rp = random_projection.GaussianRandomProjection(n_components=N_COMPONENTS)
new_data = rp.fit_transform(DATASET.training_features)

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
elif MODE == 'compare':
    success = 0
    for i in range(1000):
        if i % 50 == 0:
            print("{}/{}".format(i, 1000))
        rp = random_projection.GaussianRandomProjection(n_components=N_COMPONENTS)
        new_data = rp.fit_transform(DATASET.training_features)
        kmeans = cluster.KMeans(n_clusters=N_CLUSTERS)
        kmeans.fit(new_data)
        count = {}
        for i, c in enumerate(kmeans.labels_):
            if c not in count:
                count[c] = {}
            if DATASET.training_labels[i] not in count[c]:
                count[c][DATASET.training_labels[i]] = 1
            else:
                count[c][DATASET.training_labels[i]] += 1
        
        maxima = []
        for key, value in count.items():
            maxima.append(max(value, key=value.get))
        maxima = set(maxima)
        if len(maxima) == 10:
            success += 1
    print(success)

if EXPERIMENT == 'learning':
    test_scores = np.array([])
    train_scores = np.array([])
    test_scores_rp = np.array([])
    train_scores_rp = np.array([])

    for i in range(N_REPEAT):
        if i % 50 == 0:
            print("{}/{}".format(i, N_REPEAT))
        classifier = neural_network.MLPClassifier()
        classifier_rp = neural_network.MLPClassifier()

        _, train, test = model_selection.learning_curve(
            classifier, DATASET.training_features, DATASET.training_labels,
            cv=10, train_sizes=[1.0])

        _, train_rp, test_rp = model_selection.learning_curve(
            classifier_rp, new_data, DATASET.training_labels,
            cv=10, train_sizes=[1.0])
        
        test_scores = np.append(test_scores, [test])
        train_scores = np.append(train_scores, [train])

        test_scores_rp = np.append(test_scores_rp, [test_rp])
        train_scores_rp = np.append(train_scores_rp, [train_rp])

    print("Original: Test: {}    Train: {}".format(test_scores.mean(), train_scores.mean()))
    print("RP:       Test: {}    Train: {}".format(test_scores_rp.mean(), train_scores_rp.mean()))
elif EXPERIMENT == 'compute_time':
    plot_x = []
    plot_y_scoring = []
    plot_y_fitting = []
    plot_y_scoring_pca = []
    plot_y_fitting_pca = []

    for training_fraction in np.linspace(0.1, 0.9, 10):
        training_size = int(len(new_data) * training_fraction)
        print("Computing score for training_size = {} ...".format(training_size))
        classifier = neural_network.MLPClassifier()
        classifier_pca = neural_network.MLPClassifier()

        result = model_selection.cross_validate(
                classifier, 
                DATASET.training_features[:training_size],
                DATASET.training_labels[:training_size],
                cv=10, return_train_score=True)
        result_pca = model_selection.cross_validate(
                classifier_pca, 
                new_data[:training_size],
                DATASET.training_labels[:training_size],
                cv=10, return_train_score=True)
        test_score = result['test_score']
        train_score = result['train_score']
        fit_time = result['fit_time']
        score_time = result['score_time']

        test_score_pca = result_pca['test_score']
        train_score_pca = result_pca['train_score']
        fit_time_pca = result_pca['fit_time']
        score_time_pca = result_pca['score_time']

        plot_x.append(training_size)
        plot_y_scoring.append(np.array(score_time).mean())
        plot_y_fitting.append(np.array(fit_time).mean())

        plot_y_scoring_pca.append(np.array(score_time_pca).mean())
        plot_y_fitting_pca.append(np.array(fit_time_pca).mean())
    
    plt.figure()
    plt.grid()
    plt.xlabel('Set Size')
    plt.ylabel('Compute time in seconds')
    plt.title(TITLE)
    plt.plot(plot_x, plot_y_scoring)
    plt.plot(plot_x, plot_y_fitting)
    plt.plot(plot_x, plot_y_scoring_pca)
    plt.plot(plot_x, plot_y_fitting_pca)
    plt.legend(['Scoring', 'Fitting', 'Scoring PCA', 'Fitting PCA'])
    plt.show()