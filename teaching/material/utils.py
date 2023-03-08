import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

def plot_F_and_grad_2d(F, DF, theta, xs, window=(-3, 3, -3, 3), delta=0.025):
    """
    Plot the contour level of a function F, and its gradient DF at a list of given points xs. 
    
    :param F: function from R^2 to R
    :param DF: function from R^2 to R^2
    :param xs: iterable of points in R^2
    :param window: parameter to have a decent plot. 
    :param delta: parameter for quali contour plot.
    
    Can't make it work reliably zzz... Bonus point for someone fixing this. 
    """
    
    fig, ax = plt.subplots()
    
    xmin, xmax, ymin, ymax = window
    
    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = jnp.meshgrid(x, y)
    
    def wrap_F(u,v):
        return F(jnp.array([u,v]), theta)
    
    Z = wrap_F(X, Y)
    
    CS = ax.contour(X, Y, Z)
    
    
def plot_gradient_field(DF, theta, window=(-5, 5, -5, 5), num = 15, lr = 0.1):
    """
    Plot the gradient field described by DF (with optional parameter theta). 
    
    :param DF: function from R^2 to R^2
    :param theta: optional parameter for DF if needed. 
    :param window: parameter to have a decent plot. 
    :param num: we plot num x num arrow. Don't take it too large. 
    :param lr: scaling factor for the length of the gradient. 
    
    """
    
    fig, ax = plt.subplots()
    
    xmin, xmax, ymin, ymax = window
    
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    
    for u in x:
        for v in y:
            dx = lr * DF(np.array([u,v]),theta)
            ax.scatter(u, v, marker='o', c='blue')
            ax.arrow(u, v, dx[0], dx[1], color='red')
            
            
def plot_gd_1d(F, all_x, all_losses, all_grad, lr, xs):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    ax = axs[0]
    ax.plot(xs, F(xs), c='blue', label='F')
    ax.scatter(all_x, all_losses, marker='o', c='red', label='GD')
    for (x,f,g) in zip(all_x, all_losses, all_grad):
        ax.arrow(x,f, - lr * g, 0, color='red', width = min(0.01, 0.1 * np.abs(lr * g)), length_includes_head=True)
    ax.grid()
    ax.legend(fontsize=16)
    
    ax.set_title("A one dimensional Gradient Descent", fontsize=18)
    
    ax = axs[1]
    ax.plot(all_losses)
    ax.grid()
    ax.set_title("Evolution of $F(x)$ over iterations", fontsize=18)
    

def data_generation():
    a, b = 3.3, -3.1

    def true_f(x):
        return x * jnp.exp(- (x - b)**2 /2 ) + jnp.exp(- (x - a)**2 / 2)

    n = 500
    x_train = np.random.uniform(low=-10, high=10, size=n)
    y_train = true_f(x_train) + 0.1 * np.random.randn(n)
    
    return x_train, y_train


def plot_sgd(F, x_train, y_train, thetas):
    fig, axs = plt.subplots(1, 3, figsize=(20,6))
    
    thetas = np.array(thetas)
    
    ax = axs[0]
    
    ax.plot(thetas[:,0], thetas[:,1], label=r'$\theta$ updates', zorder=1)
    #ax.scatter(true_theta[0], true_theta[1], c='red', s=200, label='ground truth', zorder=2)
    ax.scatter(thetas[0,0], thetas[0,1], c='black', s=200, label='starting point', zorder=2)
    ax.scatter(thetas[-1][0], thetas[-1][1], c='blue', s=200, label='final point', zorder=2)
    ax.grid()
    ax.legend()

    xs = np.linspace(-10, 10, 500)
    
    ax = axs[1]
    ax.scatter(x_train, y_train, label="training set")
    ax.plot(xs, F(xs, thetas[0]), c='red', linewidth=3)
    ax.set_title(r"Initial model (with $\theta_0$)", fontsize=18)
    ax.grid()
    ax.legend()
    
    ax = axs[2]
    ax.scatter(x_train, y_train, label="training set")
    ax.plot(xs, F(xs, thetas[-1]), c='red', linewidth=3)
    ax.set_title(r"Final model (with $\theta_T$)", fontsize=18)
    ax.grid()
    ax.legend()
    
    
    
def generate_data_regression():
    n = 15
    x_train = np.random.uniform(low=-3.5, high=3.5, size=n)
    x_test = np.random.uniform(low=-3, high=3, size=n)

    def P(x):
        Px =  1 - 3*x**2 + 0.25 * x**4 + 0.5 * np.random.randn(n)
        Px[0] += 10
        return Px
    
    y_train = P(x_train)
    y_test = P(x_test)
    
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, label="train data")
    ax.scatter(x_test, y_test, label='test data')
    ax.legend()
    ax.grid()
    
    return x_train, y_train, x_test, y_test


def display_polynom(model):
    print("Our model encode the following polynomial expression:")
    theta = np.round(model.coef_, 2)
    for i,a in enumerate(theta):
        if i < len(theta)-1:
            print(a, "X^%s" %(i), " + ", end='')
        else:
            print(a, "X^%s" %(i))
