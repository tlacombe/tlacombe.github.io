import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import gudhi as gd
from gudhi.representations import Landscape, PersistenceImage, Silhouette
from gudhi.point_cloud.timedelay import TimeDelayEmbedding

def plot_circle_around_pts_cloud(X, r, ax=None):
    '''
    Plot circles around a (2D) point cloud X, with radius r. 

    Useful to visualise the Rips filtration. 
    '''
    if ax is None:
        fig, ax = plt.subplots()

    for x in X:
        c = Circle(x, r, alpha=0.2)
        ax.add_patch(c)


def sample_circle(n, r, eps):
    '''
    Sample n points on a circle of radius r, 
    plus a gaussian noise of variance eps. 
    '''
    thetas = 2 * np.pi * np.random.rand(n)
    X = r * np.array([np.sin(thetas), np.cos(thetas)]).T
    X = X + eps * np.random.randn(n, 2)
    return X


def sample_torus(n, r1=1, r2=0.2):
    # Sample points uniformly on a torus of big radius r1 and small radius r2
    theta1 = 2 * np.pi * np.random.rand(n)
    theta2 = 2 * np.pi * np.random.rand(n)

    x = (r1 + r2 * np.cos(theta2)) * np.cos(theta1)
    y = (r1 + r2 * np.cos(theta2)) * np.sin(theta1)
    z = r2 * np.sin(theta2)

    X = np.array([x, y, z]).T

    return X


def CechDiagram(X, homology_dimension=None):
    st = gd.AlphaComplex(points=X).create_simplex_tree()
    dgm = st.persistence(min_persistence=0.001)
    if homology_dimension is not None:
        dgm = np.sqrt(np.array(st.persistence_intervals_in_dimension(homology_dimension)))
    else:
        dgm = [(dim, np.sqrt(coord)) for dim, coord in dgm]
    return dgm  # The sqrt is an artefact due to the use of the AlphaComplex


def RipsDiagram(X, homology_dimension=None, max_dimension=2):
    st = gd.RipsComplex(points=X).create_simplex_tree(max_dimension=max_dimension)
    dgm = st.persistence()
    if homology_dimension is not None:
        dgm = np.array(st.persistence_intervals_in_dimension(homology_dimension))
    return dgm


def generate_orbit(num_pts, r):
    X = np.empty([num_pts,2])
    x, y = np.random.uniform(), np.random.uniform()
    for i in range(num_pts):
        X[i,:] = [x, y]
        x = (X[i,0] + r * X[i,1] * (1-X[i,1])) % 1.
        y = (X[i,1] + r * x * (1-x)) % 1.
    return X


#####################

def load_data_motions(idx=0):
    data = np.load('./data/data_v2.npy')

    s = 3
    fs = 10, 10
    fig = plt.figure(figsize=fs)
    ax = fig.add_subplot(projection='3d')
    for S, label in zip(data, ['walking', 'stepper', 'cross', 'jumping']):
        ax.scatter(S[idx][:,0], S[idx][:,1], S[idx][:,2], s=s, label=label)
    ax.legend(fontsize=fs[0]*2)
    
    return data




######################
# Illustrative plots #
######################

def showcase_vectorization():
    num_pts = 1000
    r = 3.5
    X = generate_orbit(num_pts, r)
    dgmX = CechDiagram(X, homology_dimension=1)

    fs = 20, 4
    fig, axs = plt.subplots(1, 4, figsize=fs)

    ax = axs[0]
    ax.scatter(X[:,0], X[:,1])
    ax.grid()

    ax = axs[1]
    gd.plot_persistence_diagram(dgmX, axes=ax)
    ax.set_title("Persistence diagram")

    ax = axs[2]
    reso = 100, 100
    PI = PersistenceImage(bandwidth=2*1e-3, weight=lambda x: x[1]**2, \
                                         im_range=[0,.06,0,.06], resolution=reso)
    pi = PI.fit_transform([dgmX])

    ax.imshow(np.flip(np.reshape(pi[0], reso), 0))
    ax.set_title("Persistence Image")



    ax = axs[3]
    SH = Silhouette(resolution=1000, weight=lambda x: np.power(x[1]-x[0],1))
    sh = SH.fit_transform([dgmX])
    ax.plot(sh[0])
    ax.set_title("Persistence silhouette")


def showcase_time_delay_embedding():

    n = 100

    def s(t):
        return np.sin(t) + np.cos(t**2)
    
    ts = np.linspace(0, 5, n)

    S = s(ts)
    #W = np.array([S[:-1], S[1:]]).T
    W = TimeDelayEmbedding(dim=2,delay=1,skip=1)(S)

    idx = 80

    fs = 12, 4
    fig, axs = plt.subplots(1, 2, figsize=fs)
    ax = axs[0]
    ax.plot(ts, S)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$s(t)$")
    ax.set_title("Signal")
    ax.scatter(ts[idx:idx+2], S[idx:idx+2], s=50, marker='x', color='black', label='ref')
    ax.grid()
    
    ax = axs[1]
    ax.scatter(W[:,0], W[:,1], color='red')
    ax.set_title("Time Delay embedding, $k=2$")
    ax.scatter(W[idx,0], W[idx,1], marker='x', s=50, color='black', label='corresp. pt')

    [ax.legend() for ax in axs]


def showcase_sw_periodicity(noise_level=1.):
    n = 200

    def s(t):
        return np.sin( 50 * t + np.exp(2*t) * (t > 1)) + np.arctan(100 * (t - 1))
    
    ts = np.linspace(0, 2, n)

    S = s(ts) + noise_level / (2 * np.sqrt(n)) * np.random.randn(n)
    W = np.array([S[:-1], S[1:]]).T

    dgm = CechDiagram(W)

    idx = int(0.5 * n)

    fs = 16, 4
    fig, axs = plt.subplots(1, 3, figsize=fs)
    ax = axs[0]
    ax.plot(ts, S)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$s(t)$")
    ax.set_title("Signal")
    #ax.scatter(ts[idx:idx+2], S[idx:idx+2], s=50, marker='x', color='black', label='ref')
    ax.grid()
    
    ax = axs[1]
    ax.scatter(W[:,0], W[:,1], color='red')
    ax.set_title("Time Delay embedding, $k=2$")
    ax.set_aspect('equal')
    #ax.scatter(W[idx,0], W[idx,1], marker='x', s=50, color='black', label='corresp. pt')

    ax = axs[2]
    gd.plot_persistence_diagram(dgm, axes=ax)

def showcase_cech_simplicial_filtration():
    n = 3
    thetas = 2 * np.pi * np.linspace(0,1, n, endpoint=False)
    X = np.array([np.sin(thetas), np.cos(thetas)]).T

    fs = 12, 3
    fig, axs = plt.subplots(1, 2, figsize = fs)
    for ax in axs:
        ax.scatter(X[:,0], X[:,1], c='red')
        ax.plot([*X[:,0],X[0,0]], [*X[:,1], X[0,1]], color='blue')
        ax.set_axis_off()
        ax.set_aspect('equal')

    r1 = np.pi / n - 0.1
    ax = axs[0]
    plot_circle_around_pts_cloud(X, r1, ax=ax)

    r2 = np.pi / n + 0.1
    ax = axs[1]
    plot_circle_around_pts_cloud(X, r2, ax=ax)
    ax.add_patch(Polygon([*X[:3],X[0]], fill=True, color='lightgrey', label='Faces', alpha=0.7))


def showcase_rips_simplicial_filtration():

    n = 3
    thetas = 2 * np.pi * np.linspace(0,1, n, endpoint=False)
    X = np.array([np.sin(thetas), np.cos(thetas)]).T

    fs = 12, 6
    fig, axs = plt.subplots(2, 2, figsize = fs)
    for ax in axs[0]:
        ax.scatter(X[:,0], X[:,1], c='red')
        ax.plot([*X[:,0],X[0,0]], [*X[:,1], X[0,1]], color='blue')
        ax.set_axis_off()
        ax.set_aspect('equal')

    r1 = np.pi / n - 0.1
    ax = axs[0,0]
    plot_circle_around_pts_cloud(X, r1, ax=ax)
    ax.add_patch(Polygon([*X[:3],X[0]], fill=True, color='lightgrey', label='Faces', alpha=0.7))
    
    r2 = np.pi / n + 0.1
    ax = axs[0,1]
    plot_circle_around_pts_cloud(X, r2, ax=ax)
    ax.add_patch(Polygon([*X[:3],X[0]], fill=True, color='lightgrey', label='Faces', alpha=0.7))

    
    n = 4
    thetas = 2 * np.pi * np.linspace(0,1, n, endpoint=False)
    X = np.array([np.sin(thetas), np.cos(thetas)]).T


    for ax in axs[1]:
        ax.scatter(X[:,0], X[:,1], c='red')
        ax.plot([*X[:,0],X[0,0]], [*X[:,1], X[0,1]], color='blue')
        ax.set_axis_off()
        ax.set_aspect('equal')

    r1 = np.pi / n
    ax = axs[1,0]
    plot_circle_around_pts_cloud(X, r1, ax=ax)

    r2 = 2 * np.pi / n
    ax = axs[1,1]
    plot_circle_around_pts_cloud(X, r2, ax=ax)
    ax.plot([X[0,0], X[2,0]], [X[0,1], X[2,1]], color='blue')
    ax.plot([X[1,0], X[3,0]], [X[1,1], X[3,1]], color='blue')
    ax.add_patch(Polygon([*X[:n],X[0]], fill=True, color='lightgrey', label='Faces', alpha=0.7))


def showcase_simplicial_filtration():
    '''
    Todo, can we do something more dynamic with true values from a random function ? (e.g. lower star filtration)
    '''
    n = 4
    m = 5
    thetas = 2 * np.pi * np.linspace(0,1, n, endpoint=False)
    X = np.array([np.sin(thetas), np.cos(thetas)]).T

    fs = 4*m,3
    fig, axs = plt.subplots(1,m, figsize=fs)

    ax = axs[0]
    ax.scatter(X[:,0], X[:,1], color='red', label='Vertices', zorder=3)
    ax.plot([*X[:,0],X[0,0]], [*X[:,1], X[0,1]], color='blue', label='Edges')
    ax.plot([X[2,0], X[0,0]], [X[2,1], X[0,1]], color='blue')
    ax.add_patch(Polygon([*X[:3],X[0]], fill=True, color='lightgrey', label='Faces'))
    for x in X:
        ax.annotate('0', 1.1 * x, color='red', ha='center', va='center')
    for i in range(n):
        ax.annotate('1', 1.1 * (X[i] + X[(i+1) % n])/2, ha = 'center', va='center', color='blue')
    ax.annotate('2', 1.1 * (X[2] + X[0])/2 - (0.1, 0), va='center', color='blue')
    ax.annotate('3', 1.1 * (X[2] + X[1] + X[0])/3, va='center', color='black')
    
    #ax.legend()
    
    ax.set_title("Simplicial Complex and $f$ values")

    for i, ax in enumerate(axs):
        ax.scatter(X[:,0], X[:,1], color='red', label='Vertices', zorder=3)
        ax.set_axis_off()
        ax.set_aspect('equal')
        if i > 0:
            ax.set_title("$S_{%s}$"%(i-1))

    for ax in axs[2:]:
        ax.plot([*X[:,0],X[0,0]], [*X[:,1], X[0,1]], color='blue', label='Edges')

    for ax in axs[3:]:
        ax.plot([X[2,0], X[0,0]], [X[2,1], X[0,1]], color='blue')
    ax = axs[4]
    ax.add_patch(Polygon([*X[:3],X[0]], fill=True, color='lightgrey', label='Faces'))

def showcase_simplicial_complex():
    n = 8
    thetas = 2 * np.pi * np.linspace(0,1,n, endpoint=False) + 0.1 * np.random.randn(n) # np.random.rand(n)
    thetas.sort()
    X = np.array([np.sin(thetas), np.cos(thetas)]).T
    
    fs = 5,5
    fig, ax = plt.subplots(figsize=fs)

    ax.scatter(X[:,0], X[:,1], color='red', label='Vertices',zorder=5)
    ax.plot(X[:,0], X[:,1], color='blue', label='Edges')
    ax.plot([X[2,0], X[0,0]], [X[2,1],X[0,1]], color='blue')

    ax.plot([X[-1,0], X[0,0]], [X[-1,1],X[0,1]], color='blue')
    ax.plot([X[-1,0], X[-3,0]], [X[-1,1],X[-3,1]], color='blue')

    idx = np.random.randint(n, size=3)
    idx.sort()
    ax.plot([*X[idx,0], X[idx[0],0]], [*X[idx,1], X[idx[0],1]], color='blue')
    

    idx = np.random.randint(n, size=3)
    idx.sort()
    ax.plot([*X[idx,0], X[idx[0],0]], [*X[idx,1], X[idx[0],1]], color='blue')
    
    ax.add_patch(Polygon([*X[:3],X[0]], fill=True, color='lightgrey', label='Faces'))
    ax.add_patch(Polygon([*X[-3:],X[-3]], fill=True, color='lightgrey'))
    ax.add_patch(Polygon([*X[idx],X[idx[0]]], fill=True, color='lightgrey'))
    
    ax.legend()
    ax.set_axis_off()
    ax.set_title("A (2-dimensional) simplicial complex")


def showcase_barcode_and_dgm():
    X = sample_torus(1000, r1 = 5, r2=2)
    dgm = CechDiagram(X)
    
    fig = plt.figure(figsize=(15,2.8))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:,0], X[:,1], X[:,2], alpha=0.1, color='red')
    ax1.set_zlim(-5, 5)
    ax1.set_axis_off()
    ax1.set_title("Input point cloud (torus)")
    ax1.view_init(elev=40)

    ax2 = fig.add_subplot(132)
    gd.plot_persistence_barcode(dgm, axes=ax2)

    ax3 = fig.add_subplot(133)
    gd.plot_persistence_diagram(dgm, axes=ax3)

def showcase_tda_stability():
    n = 300
    eps = 0.05
    X1 = sample_circle(n, 1, 0)
    X2 = X1 + eps * np.random.randn(n, 2)

    d1 = CechDiagram(X1)
    d2 = CechDiagram(X2)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    ax = axs[0,0]
    ax.scatter(X1[:,0], X1[:,1])
    ax.set_axis_off()
    ax.set_title("Point cloud")

    gd.plot_persistence_diagram(d1, axes=axs[1,0])

    ax = axs[0, 1]
    ax.scatter(X2[:,0], X2[:,1])
    ax.set_axis_off()
    ax.set_title("Perturbed point cloud")

    gd.plot_persistence_diagram(d2, axes=axs[1,1])


def showcase_Cech_sublevelset(parameter):
    n = 100
    X = sample_circle(n, 1, 0)
    fs = (5,5)

    thetas = 2 * np.pi * np.linspace(0, 1, 100)

    custom_circle = Circle(0, 1, alpha=0.2, label='Cech sublevetset, $t=%.2f$' %parameter)
    
    fig, ax = plt.subplots()
    plot_circle_around_pts_cloud(X, parameter, ax=ax)
    l1 = ax.plot(np.sin(thetas), np.cos(thetas), color='black', label='underlying $X$')
    l2 = ax.scatter(X[:,0], X[:,1], color='red', label='point cloud $X_n$', zorder=5)   
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    handles.append(custom_circle)
    ax.legend(handles=handles)


def showcase_birth_death():
    n = 10
    thetas = 2 * np.pi * np.linspace(0, 1, n,endpoint=False)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    ax = axs[0]
    ax.scatter(np.sin(thetas), np.cos(thetas), color='red', label='point cloud $X_n$', zorder=5)
    plot_circle_around_pts_cloud(np.array([np.sin(thetas), np.cos(thetas)]).T, r=np.pi/n, ax=ax)
    ax.set_title("The loop appears")
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    ax = axs[1]
    ax.scatter(np.sin(thetas), np.cos(thetas), color='red', label='point cloud $X_n$', zorder=5)
    plot_circle_around_pts_cloud(np.array([np.sin(thetas), np.cos(thetas)]).T, r=0.8, ax=ax)
    ax.set_title("The loop is still there")
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    ax = axs[2]
    ax.scatter(np.sin(thetas), np.cos(thetas), color='red', label='point cloud $X_n$', zorder=5)
    plot_circle_around_pts_cloud(np.array([np.sin(thetas), np.cos(thetas)]).T, r=1, ax=ax)
    ax.set_title("The loop disappears")
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    

def showcase_point_cloud_trivial_homology():
    X = sample_circle(75, 1, 0.1)
    fs = (3,3)
    fig, ax = plt.subplots(figsize=fs)
    ax.scatter(X[:,0], X[:,1])
    ax.set_axis_off()
    ax.set_title("The homology groups of this shape are trivial")


def showcase_sublevel_sets(parameter):
    if parameter <= 0:
        raise ValueError('The current implementation only allows for parameter > 0\n (this constraint is due to plotting purpose only)')
        
    n=500
    ts = np.linspace(0, 10, n)
    def f(t):
        return 1 + np.sin(np.exp(np.sqrt(t))) * np.cos(t)
    fs = (4,3)
    fig, ax = plt.subplots(figsize=fs)

    threshold = np.minimum(parameter, f(ts))
    val = f(ts)
    sublevelset = np.zeros(n)
    sublevelset[np.where(val > parameter)] = np.nan

    
    ax.plot(ts, val, label='$f$')
    ax.set_ylabel('$\mathbb{R}$')
    ax.set_xlabel('$\mathcal{X}$')
    ax.set_xticks([])
    ax.set_title('Sublevel set of $f$ with $t = %.2f$' %parameter)
    ax.fill_between(ts, threshold, color='lightgrey')
    ax.plot(ts, sublevelset, color='red', linewidth=3, label='$\mathcal{F}_t$')
    ax.legend()

def showcase_homotopy():
    thetas = 2 * np.pi * np.linspace(0,1,100)
    fs = (9, 3)
    
    fig, axs = plt.subplots(1,2, figsize=fs)
    ax = axs[0]
    ax.plot(np.sin(thetas), np.cos(thetas))
    ax.set_axis_off()

    ax = axs[1]
    ax.plot(np.sin(thetas), np.cos(thetas) * (1 - np.sin(thetas)**2))
    ax.set_axis_off()

    fig.suptitle("Two homotopic shapes")

    fig, axs = plt.subplots(1,2, figsize=fs)
    ax = axs[0]
    ax.plot(np.sin(thetas), np.cos(thetas))
    ax.set_axis_off()

    ax = axs[1]
    ax.plot(np.sin(thetas), thetas)
    ax.set_axis_off()

    fig.suptitle("Two non-homotopic shapes (tear up)")
    
    fig, axs = plt.subplots(1,2, figsize=fs)
    ax = axs[0]
    ax.plot(np.sin(thetas), np.cos(thetas))
    ax.set_axis_off()

    ax = axs[1]
    ax.plot(np.sin(thetas), np.cos(thetas) * np.sin(thetas))
    ax.set_axis_off()

    fig.suptitle("Two non-homotopic shapes (need self intersect to go from left to right)")