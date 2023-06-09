import pickle
import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from network.TwoLayer import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

def train(x_train, t_train, x_test, t_test, initParams=None):
    # Because the output is between 0-9 so we set output_size to 10
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, initParams=initParams)

    iters_num = 10000
    train_size = x_train.shape[0] # 60000
    batch_size = 100
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size) # Select a batch_size between 0 - train_size
        
        # Randomly select a part of data
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        if not initParams:
            # Update params
            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] -= learning_rate * grad[key] # Also see gradinent.py

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)


        # Only calcuate accuracy every epoch. All data passed in.
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)
    
    return network.params


dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/params.pkl"

if not os.path.exists(save_file):
    params = train(x_train, t_train, x_test, t_test)
    with open(save_file, 'wb') as f:
        pickle.dump(params, f, -1)
else:
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    test_data = np.array([x_test[1000]])
    expected_answer = np.array([t_test[1000]])

    train(x_train, t_train, test_data, expected_answer, dataset)
