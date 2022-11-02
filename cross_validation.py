#Added by Ali Abdullah
import numpy as np
from itertools import product


#using k fold cross-validation

def cross_validate(dataset, generate_function, possible_hyperparam_values,
                   # TODO: Add your own arguments if you need them!
                   num_folds
                   ):
    """
    TODO: Please provide a docstring for this function
    performs k-fold cross-validation

    Input:
    TODO: Please provide a description for every argument of the function
    generate_function - static method for creating the model with given hyperparameters
    possible_hyperparam_values - all the given hyperparameter values to train the model on 
    num_folds - number of folds/slices to divide the dataset in 

    Returns:
    best_model - an instance of the best trained model
    best_hyperparameters - a dict of hyperparameters values that was considered best based on accuracy
    """

    ########################
    
    data = dataset.X.copy()
    labels = dataset.y.copy()

    fold_size = int(len(data) / num_folds)

    indices = [i for i in range(len(data))]
    np.random.shuffle(indices)
        
    new_hyperparam_dict = {}

    results = possible_hyperparam_values.copy()
    results["model"] = []
    results["avg_accuracy"] = []

    hyperparams_values_list = []
    hyperparams_list = []
    for keys, values in possible_hyperparam_values.items():
        hyperparams_values_list.append(values)
        hyperparams_list.append(keys)

    for hyperparams_values in list(product(*hyperparams_values_list)):
        for i in range(len(hyperparams_list)):
            new_hyperparam_dict[hyperparams_list[i]] = hyperparams_values[i] 
            
        model = generate_function(new_hyperparam_dict)
        scores = []
        for i in range(num_folds):
            test_indices = indices[i*fold_size:(i+1)*fold_size]
            test_data = data[test_indices]
            test_labels = labels[test_indices]
            train_data = np.delete(data, test_indices, axis = 0)
            train_labels = np.delete(labels, test_indices, axis = 0)
            model.fit(train_data, train_labels)
            accuracy = model.score(test_data, test_labels)
            scores.append(accuracy)

        avg_accuracy = sum(scores)/len(scores)
        results["model"].append(model)
        results["avg_accuracy"].append(avg_accuracy)

    best_result_index = np.argmax(np.array(results["avg_accuracy"]))
    best_params = {}
    for key, values in results.items():
        best_params[key] = results[key][best_result_index]

    best_model = best_params["model"]

    del best_params["avg_accuracy"]
    del best_params["model"]

    return best_model, best_params
    #######################