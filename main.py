from data import *
from domain_adaptation import run_domain_adaptation
from train import *
import numpy as np
from numpy import savetxt

def mnist_demo(num_epochs=5):
    learning_curves = {}
    alpha_values = [-2, -1, 0.5, 1, 2, 10]
    train_loader, test_loader = get_MNIST_data()
    for i in range(len(alpha_values)):
        alpha = alpha_values[i]
        print("Start train with alpha = ", alpha)
        net, codes = train(train_loader, test_loader, alpha=alpha, num_epochs=num_epochs)
        print("Finish train with alpha = ", alpha)
        learning_curves[alpha] = codes['lowerbound']
        savetxt('learning_curves_alpha_{}.csv'.format(alpha), learning_curves[alpha], delimiter=',')

    print("Plot learning curves")
    plot_learning_curve(num_epochs, alpha_values)


def toy_data_demo(num_epochs=10, learning_rate=1e-6):
    num_features = 784
    num_centers = 3

    centers_base = [10, 25, 40]
    centers = np.zeros((num_centers, num_features))
    for i in range(num_centers):
        for j in range(num_features):
            centers[i][j] = centers_base[i] + np.random.rand()*10

    stds = np.random.rand(num_centers, num_features)

    num_samples = np.random.rand(num_centers)*5000
    num_samples = [int(n) for n in num_samples]

    X_train, X_test, Y_train, Y_test = generate_toy_data(centers, stds, num_samples, num_features)
    plot_toy_data(X_train, Y_train, "train_data")
    plot_toy_data(X_test, Y_test, "test_data")

    X_train = np.array([x.reshape(28, 28) for x in X_train])
    X_test = np.array([x.reshape(28, 28) for x in X_test])
    train_loader, test_loader = generate_data_loaders(X_train, X_test, Y_train, Y_test)

    # log_px_vals = []
    # for x, y in test_loader:
    #     log_px = calc_log_px(centers, stds, num_samples, x, y)
    #     log_px_vals.append(log_px)
    # log_px = np.mean(log_px_vals)

    learning_curves = {}
    alpha_values = [-2, -1, 0.5, 1, 2, 10]

    for i in range(len(alpha_values)):
        alpha = alpha_values[i]
        print("Start train with alpha = ", alpha)
        net, codes = train(train_loader, test_loader, alpha=alpha, num_epochs=num_epochs, learning_rate=learning_rate)
        print("Finish train with alpha = ", alpha)
        learning_curves[alpha] = codes['lowerbound']
        savetxt('learning_curves_alpha_{}.csv'.format(alpha), learning_curves[alpha], delimiter=',')

    print("Plot learning curves")
    plot_learning_curve(num_epochs, alpha_values)


def main():
    # toy_data_demo(num_epochs=100, learning_rate=1e-5)
    mnist_demo(num_epochs=100)
    # alpha_values = [-2, -1, 0.5, 1, 2, 10]
    # alpha_values = [1, 2, 10]
    # plot_learning_curve(100, alpha_values)
    # run_domain_adaptation()


if __name__ == "__main__":
    main()