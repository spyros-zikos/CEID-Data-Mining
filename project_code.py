import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN, KMeans, Birch
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

import time


people = []

try:
    data_path = input("Give harth file path (e.g. C:\harth\ or harth\): ")
    print("Loading dataset...")
    df = pd.read_csv(data_path + "S006.csv")
    people.append(pd.read_csv(data_path + "S006.csv"))
except:
    print("Path not valid!")
    exit()

for i in ['08','09',10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]:
    df = df.append(pd.read_csv(data_path + "S0" + str(i) + ".csv"))
    people.append(pd.read_csv(data_path + "S0" + str(i) + ".csv"))

df = df.drop("index", axis=1).drop("Unnamed: 0", axis=1).reset_index(drop=True)

def analysis():
    df.info()
    print(df.describe())
    show_hist_plots()
    show_heatmap()

def show_hist_plots():
    plt.figure('back_x distribution')
    sns.histplot(df.back_x.dropna(), bins=50, kde=False)
    plt.figure('back_y distribution')
    sns.histplot(df.back_y.dropna(), bins=50, kde=False)
    plt.figure('back_z distribution')
    sns.histplot(df.back_z.dropna(), bins=50, kde=False)
    plt.figure('thigh_x distribution')
    sns.histplot(df.thigh_x.dropna(), bins=50, kde=False)
    plt.figure('thigh_y distribution')
    sns.histplot(df.thigh_y.dropna(), bins=50, kde=False)
    plt.figure('thigh_z distribution')
    sns.histplot(df.thigh_z.dropna(), bins=50, kde=False)
    plt.figure('label distribution')
    sns.histplot(df.label.dropna(), bins=50, kde=False)
    plt.show()

def show_heatmap():
    corr = df.corr()
    plt.figure(figsize=(7,7))
    sns.heatmap(corr, cmap="Greens", annot=True)
    plt.show()

##### Classification #####
def select_classifier(option):
    if option == 'Multi-layer Perceptron':
        classifier = MLPClassifier(max_iter=10000, activation='logistic', random_state=42)
    elif option == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    elif option == 'Naive Bayes':
        classifier = GaussianNB()
    return classifier

def run_classification(X,Y,classifier):
    # not very good because it uses random spliting and we want to use correlation between neighbour samples
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = select_classifier(classifier)
    model.fit(X_train,y_train)
    model.predict(X_train)
    predictions_test = model.predict(X_test)
    print("---------------------------------------------------------")
    print(f"Classifier: {classifier}\n\tTraining accuracy: {model.score(X_train,y_train)}\n\tTesting accuracy: {accuracy_score(y_test, predictions_test)}\n")
    print(classification_report(y_test,predictions_test))

def classifiers():
    data = df.drop("label", axis=1).drop("timestamp", axis=1)
    new_data = {'back_x_m1': data["back_x"][1:-1].reset_index(drop=True), 'back_y_m1': data["back_y"][1:-1].reset_index(drop=True), 'back_z_m1': data["back_z"][1:-1].reset_index(drop=True),
                'thigh_x_m1': data["thigh_x"][1:-1].reset_index(drop=True), 'thigh_y_m1': data["thigh_y"][1:-1].reset_index(drop=True), 'thigh_z_m1': data["thigh_z"][1:-1].reset_index(drop=True),
                'back_x_m2': data["back_x"][:-2].reset_index(drop=True), 'back_y_m2': data["back_y"][:-2].reset_index(drop=True), 'back_z_m2': data["back_z"][:-2].reset_index(drop=True),
                'thigh_x_m2': data["thigh_x"][:-2].reset_index(drop=True), 'thigh_y_m2': data["thigh_y"][:-2].reset_index(drop=True), 'thigh_z_m2': data["thigh_z"][:-2].reset_index(drop=True)}
    X = data[2:].reset_index(drop=True)
    X = X.assign(**new_data)

    Y = df["label"][2:].reset_index(drop=True)
    models = ["Multi-layer Perceptron", "Random Forest", "Naive Bayes"]

    for model in models:
        start_time = time.time()
        run_classification(X, Y, model)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\tElapsed time: ", elapsed_time)

##### Clustering #####
def select_clusterer(option):
    if option == 'KMeans':
        clusterer = KMeans(n_clusters=3, random_state=0)
    elif option == 'DBSCAN':
        clusterer = DBSCAN(eps=3.9, min_samples=2)
    elif option == 'Birch':
        clusterer = Birch(threshold=3.3, n_clusters=None)
    return clusterer

def run_clusterer(X,clusterer):
    model = select_clusterer(clusterer)
    md = model.fit(X)
    labels = md.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"Clusterer: {clusterer}")
    print("\tNumber of clusters: %d" % n_clusters_)
    print("\tNumber of noise points: %d" % n_noise_)
    print(f"\tSilhouette Coefficient: {silhouette_score(X, labels):.3f}")
    print(f"\tLabels: {labels}")

def clusterers():
    simple_people = [ x.drop("label", axis=1).drop("timestamp", axis=1)
                   .drop("index", axis=1, errors="ignore").drop("Unnamed: 0", axis=1, errors="ignore")
                   .reset_index(drop=True) for x in people ]
    stats_of_people = [ list(x.mean())+list(x.std())+list(x.max())+list(x.min()) for x in simple_people ]
    X = stats_of_people

    models = ["KMeans", "DBSCAN", "Birch"]

    for model in models:
        start_time = time.time()
        run_clusterer(X, model)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\tElapsed time: ", elapsed_time)


def main():

    option = input("Select '1' to see dataset stats\nSelect '2' to see classification results\nSelect '3' to see clustering results\n")
    if option == '1':
        analysis()
    elif option == '2':
        classifiers()
    elif option == '3':
        clusterers()
    else:
        exit()

main()
