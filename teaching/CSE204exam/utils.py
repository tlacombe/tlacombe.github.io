import numpy as np
import matplotlib.pyplot as plt

def test_grad(grad_E):

	xs = np.load("data/x_train_2d.npy")
	ys = np.load("data/y_train_2d.npy")
	thetas = np.load(...)

	res = np.load(...)

	e = 0.

	for X, y, theta, res in zip(xs, ys, thetas, solutions):
		e = e + np.linalg.norm(grad_E(X,y,theta) - res)

	if e < 1E-5:
		print("Test passed, congrats!")
	else:
		print("Test failed. Error between your function and the true value of the gradient:", e, " (should be close to 0).")


def test_1d_logreg(grad_descent, pred, sigma):
	np.random.seed(seed=42)
	n = 100
	
	np.random.seed(0)
	x = np.random.normal(size=n)
	y = (x > 0).astype(np.float)
	x[x > 0] *= 4
	x += .3 * np.random.normal(size=n)
	x = np.array([x]).T

	theta, evol = grad_descent(x, y, eta=0.1, nb_max_step=100, stopping_criterion=0.)
	hat_y = [pred(xi, theta) for xi in x]

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	cmap = ["red" if yi==1 else "blue" for yi in hat_y ]
	
	ax1.scatter(x, y, color=cmap)
	
	z = np.linspace(-5, 10, 300)
	zz = [sigma(zi, theta) for zi in z]
	ax1.plot(z,zz, color="black")


	ax2.plot(evol, label="E(t)", color="blue")
	plt.show()


def plot_linreg_errors(train_errors, test_errors):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(train_errors, color="blue", label="train errors")
	ax.plot(test_errors, color="red", label="test errors")
	ax.set_title("evolution of train and test errors wrt k")
	ax.set_xlabel("k")
	ax.set_ylabel("error")
	ax.legend()

### LOAD DATASETS FUNCTIONS ###

def load_2D_set():
	x_train = np.load("data/x_train_2d.npy")
	x_test = np.load("data/x_test_2d.npy")
	y_train = np.load("data/y_train_2d.npy")
	y_test = np.load("data/y_test_2d.npy")

	cmap = ["red" if yi==1 else "blue" for yi in y_train ]
	plt.scatter(x_train[:,0], x_train[:,1],color=cmap, marker='x')
	plt.title("train set")

	return x_train, y_train, x_test, y_test

def load_kNN_dataset():
	x_train = np.load("data/x_train_NN.npy")
	x_test = np.load("data/x_test_NN.npy")
	y_train = np.load("data/y_train_NN.npy")
	y_test = np.load("data/y_test_NN.npy")

	def f(x):
		if x == 0:
			return "blue"
		if x == 1:
			return "red"
		if x == 2:
			return "green"

	cmap = [f(yi) for yi in y_train]
	plt.scatter(x_train[:,0], x_train[:,1],color=cmap, marker='x')
	plt.title("train set")

	return x_train, y_train, x_test, y_test

def load_data_polyreg():
	x_train = np.load("data/x_train_linreg.npy")
	x_test = np.load("data/x_test_linreg.npy")
	y_train = np.load("data/y_train_linreg.npy")
	y_test = np.load("data/y_test_linreg.npy")

	plt.xlabel("x")
	plt.ylabel("y")

	plt.scatter(x_train, y_train)
	plt.title("train set")

	return x_train, y_train, x_test, y_test
