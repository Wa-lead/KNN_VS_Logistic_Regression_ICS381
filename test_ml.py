if __name__ == "__main__":
    import numpy as np
    from machine_learning import *
    
    # read given datasets
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_classification_dataset()
    
    # print('X_train={}, X_val={}, X_test={}'.format(X_train.shape, X_val.shape, X_test.shape))
    # print('y_train={}, y_val={}, y_test={}'.format(y_train.shape, y_val.shape, y_test.shape))
        
    print('_______________________________________________________________________\n')
    
    # tests knn
    np.random.seed(35171)
    x_new = np.random.rand(X_train.shape[1]) # randomly generate new instance
    y_new_pred = knn_classification(X_train, y_train, x_new, k=1)
    print('x_new={}'.format(x_new))
    print('y_new_pred={}'.format(y_new_pred))
    
    print('_______________________________________________________________________\n')
    np.random.seed(1713)
    x_new = np.random.rand(X_train.shape[1]) # randomly generate new instance
    y_new_pred = knn_classification(X_train, y_train, x_new, k=5)
    print('x_new={}'.format(x_new))
    print('y_new_pred={}'.format(y_new_pred))
    
    # test logistic regression training
    trained_weights = logistic_regression_training(X_train, y_train, alpha=0.01, max_iters=1000, random_seed=1)
    print('trained_weights={}'.format(trained_weights))
    print('_______________________________________________________________________\n')
    
    trained_weights = logistic_regression_training(X_train, y_train, alpha=0.1, max_iters=500, random_seed=7171)
    print('trained_weights={}'.format(trained_weights))
    print('_______________________________________________________________________\n')
    
    # test logistic regression prediction
    train_preds = logistic_regression_prediction(X_train, trained_weights, threshold=0.5)
    print('train_preds={}'.format(train_preds[:10]))
    print('train accuracy={}'.format((y_train.flatten() == train_preds.flatten()).sum()/y_train.shape[0]))
    print('_______________________________________________________________________\n')
    
    # # test model selection and evaluation
    best_method, val_accuracy_list, test_accuracy = model_selection_and_evaluation(alpha=0.0001, max_iters=10, random_seed=321, threshold=0.7)
    print('best_method={}'.format(best_method))
    print('val_accuracy_list={}'.format(val_accuracy_list))
    print('test_accuracy={}'.format(test_accuracy))
    print('_______________________________________________________________________\n')