from data import *
from train import *
import numpy as np
from numpy import savetxt

def mnist_demo(num_epochs=5):
    learning_curves = {}
    alpha_values = [-2, -1, 0.5, 1, 2, 10]

    for i in range(len(alpha_values)):
        alpha = alpha_values[i]
        print("Start train with alpha = ", alpha)
        train_loader, test_loader = get_MNIST_data(i)
        net, codes = train(train_loader, test_loader, alpha=alpha, num_epochs=num_epochs)
        print("Finish train with alpha = ", alpha)
        learning_curves[alpha] = codes['lowerbound']
        savetxt('learning_curves_alpha_{}.csv'.format(alpha), learning_curves[alpha], delimiter=',')

    print("Plot learning curves")
    plot_learning_curve(num_epochs, learning_curves)


def toy_data_demo(num_epochs=30):
    num_features = 784
    num_centers = 3

    centers_base = np.random.rand(num_centers)
    centers = np.zeros((num_centers, num_features))
    for i in range(num_centers):
        for j in range(num_features):
            centers[i][j] = centers_base[i] + np.random.rand()*1e-5

    stds_base = np.random.rand(num_centers)
    stds = np.zeros((num_centers, num_features))
    for i in range(num_centers):
        for j in range(num_features):
            stds[i][j] = stds_base[i] + np.random.rand()*1e-10

    # centers = np.random.rand(num_centers, num_features)
    # stds = np.random.rand(num_centers, num_features)

    num_samples = np.random.rand(num_centers)*2000
    num_samples = [int(n) for n in num_samples]

    X_train, X_test, Y_train, Y_test = generate_toy_data(centers, stds, num_samples, num_features)
    plot_toy_data(X_train, Y_train, "train_data")
    plot_toy_data(X_test, Y_test, "test_data")

    X_train = np.array([x.reshape(1, 28, 28) for x in X_train])
    X_test = np.array([x.reshape(1, 28, 28) for x in X_test])
    train_loader, test_loader = generate_data_loaders(X_train, X_test, Y_train, Y_test)

    # log_px_vals = []
    # for x, y in test_loader:
    #     log_px = calc_log_px(centers, stds, num_samples, x, y)
    #     log_px_vals.append(log_px)
    # log_px = np.mean(log_px_vals)

    net, codes = train(train_loader, test_loader, alpha=0.5, num_epochs=num_epochs)


def main():
    #toy_data_demo()
    mnist_demo(num_epochs=5)

if __name__ == "__main__":
    main()