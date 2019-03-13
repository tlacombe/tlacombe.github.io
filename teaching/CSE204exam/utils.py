import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score


def score(y_true, y_pred):
	return np.round(100 * accuracy_score(y_true, y_pred), 2)


def theta_init(p):
	np.random.seed(seed=42)  # DO NOT change this
	theta = np.random.rand(p+1)
	return theta


def init_theta(p):
	np.random.seed(seed=42)  # DO NOT change this
	theta = np.random.rand(p+1)
	return theta


def test_1d_logreg(grad_descent, pred, sigma):
	np.random.seed(seed=42)
	n = 100
	
	np.random.seed(0)
	x = np.random.normal(size=n)
	y = (x > 0).astype(np.float)
	x[x > 0] *= 4
	x += .3 * np.random.normal(size=n)
	x = np.array([x]).T

	theta, evol = grad_descent(x, y, eta=0.1, nb_max_step=1000, stopping_criterion=0.00001, verbose=True)
	hat_y = [pred(xi, theta) for xi in x]

	fig = plt.figure(figsize=(16,8))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	cmap = ["red" if yi==1 else "blue" for yi in hat_y ]
	
	ax1.scatter(x, y, color=cmap)
	
	z = np.linspace(-5, 10, 300)
	zz = [sigma(zi, theta) for zi in z]
	ax1.plot(z,zz, color="black")
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	ax1.plot(z, 1/2 * np.ones(len(z)), linestyle="dashed", color="black")
	ax1.axvline(x= - theta[0] / theta[1], color="green", linestyle="dashed", label="classification boundary")
	ax1.legend()


	ax2.plot(evol, label="E(t)", color="blue")
	ax2.set_xlabel("t")
	ax2.set_ylabel("E(t)")
	ax2.legend()
	plt.show()


def test_linear_regression(linreg, E, pred):
	n = 50
	np.random.seed(42)
	x = 10 * np.random.rand(n)
	theta = [1, 1.5]
	y = theta[0] + theta[1] * x + 0.3 * np.random.randn(n)
	X = np.array([[1, xi] for xi in x])
	theta_opt = linreg(X, y)
	print("optimal theta found:", theta_opt)
	print("ground truth theta:", theta, "(both results should be close but not necessary exact.)")
	print("training loss obtained:", np.round(E(X, y, theta_opt), 2))
	z = np.linspace(0,10,100)
	#zz = np.array([theta_opt[0] + zi * theta_opt[1] for zi in z])
	zz = np.array(theta_opt[0] + zi * theta_opt[1] for zi in z])
	zzz = np.array([theta[0] + zi * theta[1] for zi in z])
	plt.scatter(x,y, color="blue")
	plt.plot(z, zz, color="red", label="your fit")
	plt.plot(z, zzz, color="red", linestyle="dashed", label="ground truth")
	plt.legend()
	plt.title("Linear regression on a 1D dataset")
	plt.xlabel("x")
	plt.ylabel("y")
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

def test_oneNN(oneNN):
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
	x_train, y_train = load_1NN_dataset()
	fig = plt.figure(figsize=(8, 8))
	vor = Voronoi(x_train)
	x1 = x_train[:,0]
	x2 = x_train[:,1]
	x_min, x_max = x1.min()- 1, x1.max() + 1
	y_min, y_max = x2.min() - 1, x2.max() + 1
	h = 0.2
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                             np.arange(y_min, y_max, h))

	Z = np.array([oneNN(x_train, y_train, z) for z in np.c_[xx.ravel(), yy.ravel()]])
	Z = Z.reshape(xx.shape)

	fig = voronoi_plot_2d(vor, show_vertices=False, show_points=False)
	fig.set_size_inches(8, 8)
	    
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.5)


	    
	plt.scatter(x1, x2, c=y_train, cmap=cmap_bold, edgecolor='k', s=30)
	plt.xlabel(r"$x_1$", fontsize=16)
	plt.ylabel(r"$x_2$", fontsize=16)



### LOAD DATASETS FUNCTIONS ###

def load_1NN_dataset():
    x_train, y_train = np.load("data/x_train_1NN.npy"), np.load("data/y_train_1NN.npy")
    return x_train, y_train

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
	plt.xlabel("x1")
	plt.ylabel("x2")

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
