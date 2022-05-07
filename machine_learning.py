import pandas as pd
import numpy as np


def preprocess_classification_dataset():
    X_train, y_train = preprocess_train('train.csv', 'output')
    X_val, y_val = preprocess_val('val.csv', 'output')
    X_test,y_test = preprocess_test('test.csv', 'output')
    return X_train, y_train, X_val, y_val, X_test,y_test

def preprocess_train(file, output):
    train_df = pd.read_csv(file)
    train_feat_df = train_df.iloc[:, :-1]
    train_output = train_df[[output]]
    X_train = train_feat_df.values
    y_train = train_output.values
    return X_train, y_train

def preprocess_val(file, output):
    val_df = pd.read_csv(file)
    val_feat_df = val_df.iloc[:, :-1]
    val_output = val_df[[output]]
    X_val = val_feat_df.values
    y_val = val_output.values
    return X_val, y_val

def preprocess_test(file, output):
    test_df = pd.read_csv(file)
    test_feat_df = test_df.iloc[:, :-1]
    test_output = test_df[[output]]
    X_test = test_feat_df.values
    y_test = test_output.values
    return X_test, y_test

# # --------------------------------------------------------------------- Knn

def knn_classification(X_train, y_train, x_new, k=5):
    # will store the pairs in (dist, index)
    nearest = [(float('inf'), float('inf')) for _ in range(k)]
    for i in range(len(X_train)):
        furthest_neighbor = max(nearest, key=lambda x: x[0])
        curr_dist = euclideandist(X_train[i], x_new)
        # if the furthest element is furhter than the current element
        if furthest_neighbor[0] > curr_dist:
            # replace them
            nearest.remove(furthest_neighbor)
            nearest.append((curr_dist, i))

    #this gathers all the neighbors votes ( outputs ) and returns what the majority selects
    voters = []
    for pair in nearest:
        voters.append(y_train[pair[1]][0])

    return max(voters, key=lambda x: voters.count(x))

#helping method to calculate the predictions of an enire set
def knn_prediciton(k, X_target, X_train, y_train):
    knn = []
    for row in X_target:
        knn.append([knn_classification(X_train, y_train, row, k=k)])
    return np.array(knn)

#helping mehtod to calc euclidean distance
def euclideandist(xi, x_new):
    return sum([(xii - x_newi)**2 for xii, x_newi in zip(xi, x_new)])
# ------------------------------------------------------------------------ Logistic regression

def logistic_regression_training(X_train, y_train, alpha=0.01, max_iters=5000, random_seed=1):
    
    #copy the dataset to avoid manupliting the reference
    X_bias = np.hstack((np.ones((len(X_train), 1)), X_train.copy()))

    num_of_features = len(X_bias[1])

    np.random.seed(random_seed)  # for reproducibility
    #initialize random weights
    weights = np.random.normal(loc=0.0, scale=1.0, size=(num_of_features, 1))

    #apply the formula in slides
    for _ in range(max_iters):
        weights = weights - alpha * \
            X_bias.T@(dataset_sigmoid(weights, X_bias) - y_train)

    return weights

def logistic_regression_prediction(X, weights, threshold=0.5):

    #copy the dataset to avoid manupliting the reference
    X_bias = np.hstack((np.ones((len(X), 1)), X.copy()))
    y_preds = []

    for instance in X_bias:
        #if the probability p (sigmoid) excceds threshold then 1 else 0
        prediction = [float(1) if sigmoid(instance.T@weights)
                      > threshold else float(0)]
        y_preds.append(prediction)

    return np.array(y_preds)

#helping methods to calculate the sigmoid of an entire set
def dataset_sigmoid(weights, set):
    return np.array([sigmoid(row.T @ weights) for row in set])

def sigmoid(scalar):
    return 1/(1+np.exp(-1*(scalar)))
# ------------------------------------------------------------------------- Model selection

def model_selection_and_evaluation(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5):
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_classification_dataset()


    #models stored into dictionary to ese the handling
    models = {
        '1nn': [knn_prediciton(1, X_val, X_train, y_train)],
        '3nn': [knn_prediciton(3, X_val, X_train, y_train)],
        '5nn': [knn_prediciton(5, X_val, X_train, y_train)],
        'logistic regression': [logistic_regression_prediction(X_val,
                                                               logistic_regression_training(
                                                                   X_train, y_train, alpha, max_iters, random_seed),
                                                               threshold)],
            }
    #calculate the accuracy for each model
    for model in models.values():
        model.append(compute_accuracy(model[0], y_val))

    val_accuracy_list = [value[1] for value in models.values()]
    #select best model based on index 1 ( accuracy )
    best_method = max(models, key=lambda x: models[x][1])  # (model, accuracy)
    test_accuracy = test_model(alpha, max_iters, random_seed, threshold, best_method)
    
    return best_method, val_accuracy_list, test_accuracy

#helping method to test the accuracy of the model on the test set
def test_model(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5, best_method=None):
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_classification_dataset()

    X_train_val_merge = np.vstack([X_train, X_val])
    y_train_val_merge = np.vstack([y_train, y_val])

    test_accuracy = None

    #this is hardcoded for our model selection, if the best model contains nn ( knn ), then test using the knn methods
    if 'nn' in best_method:
        best_method_preds = np.array(knn_prediciton(
            #best_method[0] holds the (k) number
            int(best_method[0]), X_test, X_train_val_merge, y_train_val_merge))

        test_accuracy = compute_accuracy(best_method_preds, y_test)

    #if not knn then it is the logistic
    else:
        weights = logistic_regression_training(
            X_train_val_merge, y_train_val_merge, alpha, max_iters, random_seed)
        best_method_preds = logistic_regression_prediction(
            X_test, weights, threshold)
        test_accuracy = compute_accuracy(best_method_preds, y_test)

    return test_accuracy

#--------------------------------------------------------------------
#computes the accuracy of a prediction given a set
def compute_accuracy(y_pred, y_true):
    return (y_pred.flatten(
        ) == y_true.flatten()).sum() / y_true.shape[0]