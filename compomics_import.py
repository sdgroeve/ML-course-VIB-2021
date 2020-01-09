import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from scipy.stats import entropy
import itertools

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

# Plot the decision boundaries

def plot_ensemble(bdt,X,y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis("tight")

    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class %s" % n)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),fontsize=18,
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.grid(False)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def plot_ROC(fpr,tpr):
	plt.plot(fpr, tpr, label='ROC curve')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right")
	plt.show()

def plot_decision_boundary(clf,X,y):
	cols = X.columns.values
	X = X.values
	y = y.values
	cm = plt.cm.brg	
	cm_bright = ListedColormap([ '#0000FF','#00FF00'])
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
						 np.arange(y_min, y_max, 0.2))
	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=cm, alpha=.5)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
	plt.title("accuracy=%.2f"%(clf.score(X,y)))
	plt.xlabel(cols[0])
	plt.ylabel(cols[1])
	
	
def plot_errors(a,b,X,y):
	for xi,yi in zip(X,y):
		plt.plot([xi,xi], [yi,a*xi+b], color='red', lw=1)

def cost_function(a,b,X,y):
	return (1.0/2*len(y))*((y - (a*X+b))**2).sum()

def plot_regression_path(X,y,alpha,iterations):
	for theta1 in [0.1,0.4,0.7,1,1.1,1.4,1.7,2]:
		cost = cost_function(theta1,0,X,y)
		plt.scatter(theta1,cost)
	X_list = []
	y_list = []
	theta1 = 0.0
	for i in range(iterations):
		next = np.random.randint(len(X))
		predict = np.dot(X[next],theta1)
		X_list.append(theta1)
		y_list.append(cost_function(theta1,0,X,y))
		error = predict-y[next]
		theta1 = theta1 - alpha*error*X[next]
	plt.scatter(X_list,y_list,c="r",s=70)
	plt.plot(X_list,y_list,c="r")
	plt.xlabel('theta1')
	plt.ylabel('J')
	plt.show()		
	return theta1

def plot_logistic(theta):
	f = lambda x, theta: 1. / (1. + np.exp(-x.dot(theta)))
	#create a dataset with m=2
	x = np.c_[np.ones(100), np.linspace(-10,20,100)] #x0 is set 1
	#compute f(x,theta)
	y = f(x, theta)
	plt.xlabel(r"$\theta^{\prime} x$", fontsize=20)
	plt.ylabel(r"$lr(x,\theta)$", fontsize=20)
	plt.plot(x.dot(theta), y)

def plot_lr_cost():
	a = np.arange(-6, 6, .001)

	plt.figure(figsize=(8,4))
	plt.plot(a,-1*np.log(1/(1+(np.exp(-1*a)))),linewidth=3.0,label="y=1")
	plt.plot(a,-1*np.log(1-(1/(1+(np.exp(-1*a))))),linewidth=3.0,label="y=0")
	plt.xlabel(r"$f(x,\theta)$", fontsize=22)
	plt.ylabel(r"$cost(f(x,\theta))$", fontsize=20)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def plot_lr_cost2():
	a = np.arange(-6, 6, .001)

	plt.figure(figsize=(8,4))
	plt.plot(a,-1*np.log(1/(1+(np.exp(-1*a)))),linewidth=3.0,label="y=1")
	plt.plot(a,-1*np.log(1-(1/(1+(np.exp(-1*a))))),linewidth=3.0,label="y=0")
	plt.xlabel(r"$lr(x,\theta)$", fontsize=22)
	plt.ylabel(r"$J(lr(x,\theta))$", fontsize=20)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	
	
def plot_svm_cost():
	a = np.arange(-6, 6, .001)

	k1 = lambda x: np.maximum(0, 1 - x)
	k0 = lambda x: np.maximum(0, 1 + x)


	plt.figure(figsize=(8,4))
	plt.plot(a,-1*np.log(1/(1+(np.exp(-1*a)))),linewidth=2.0, color='0.8',label="")
	plt.plot(a,-1*np.log(1-(1/(1+(np.exp(-1*a))))),linewidth=2.0, color='0.8', label="")
	plt.plot(a,k1(a),linewidth=3.0,label="y=1")
	plt.plot(a,k0(a),linewidth=3.0,label="y=0")
	plt.xlabel(r"$f(x,\theta)$", fontsize=22)
	plt.ylabel(r"$J(f(x,\theta))$", fontsize=20)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def plot_svm_decision_function(clf):
	"""Plot the decision function for a 2D SVC"""
	x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
	y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
	Y, X = np.meshgrid(y, x)
	P = np.zeros_like(X)
	for i, xi in enumerate(x):
		for j, yj in enumerate(y):
			P[i, j] = clf.decision_function(np.array([xi, yj]).reshape(1,-1))
	return plt.contour(X, Y, P, colors='k',
					   levels=[-1, 0, 1],
					   linestyles=['--', '-', '--'])

def plot_regression_degree(dataset,model,true_fun,degrees):	
	nrows  = len(degrees)/2.
	if int(nrows) < nrows:
		nrows = int(nrows)+1
	else:
		nrows = int(nrows)

	plt.figure(figsize=(12, 4*nrows))
	
	for i,d in enumerate(degrees):
		plt.subplot(nrows, 2, i+1)
		plt.subplots_adjust(hspace=.5)
		X = dataset.copy()
		X_test = pd.DataFrame(np.linspace(0, 1, 100),columns=['x1'])
		y = X.pop('y')
		for j in range(2,d+1):
			X['x1^'+str(j)] = X['x1']**j
			X_test['x1^'+str(j)] = X_test['x1']**j
		model.fit(X, y)
		pred = model.predict(X)
		pred_test = model.predict(X_test)
		plt.plot(X_test['x1'], pred_test, label="Model",lw=3)
		plt.plot(X_test['x1'], true_fun(X_test['x1']), label="True function")
		plt.scatter(X['x1'], y, label="Samples")
		plt.xlabel("x1")
		plt.ylabel("y")
		plt.xlim((0, 1))
		plt.ylim((-2, 6))
		if i == 0: plt.legend(loc="upper right")
		plt.title("degree %d (%.3f)" % (d,metrics.r2_score(y,pred)))
	plt.show()
	
def plot_linear_regression(simple):
	plt.figure(figsize=(16,10))
	for i,a in enumerate([0.1,0.5,1,1.5,2,2.5]):
		plt.subplot(2,3,i+1)
		plt.scatter(simple['x1'],simple['y'])
		plt.plot(simple['x1'], a*simple['x1'], color='green',linewidth=2)
		plt.title('a='+str(a))

def plot_entropy_function():
	pvals = np.linspace(0, 1)        
	plt.plot(pvals, [entropy([p,1-p]) for p in pvals])
	plt.xlabel(r'$p_1=1-p_2$',fontsize=18)
	plt.ylabel('entropy')

def plot_scatter_annotated(dataset,labels):
	X = dataset.values
	plt.scatter(X[:,0], X[:,1],s=100)
	
	for i, (x, y) in enumerate(zip(X[:, 0], X[:, 1])):
	    plt.annotate(
	        labels[i], 
	        xy = (x, y), xytext = (-20, 20),
	        textcoords = 'offset points', ha = 'right', va = 'bottom',
	        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5),
	        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))	
	
def plot_eigenvectors(dataset,w,v):
	plt.plot(dataset['x1'],dataset['x2'],'bo',markersize=5)
	plt.arrow(0,0,10*v[0,0],10*v[1,0],color='r',linewidth=2,head_width=1,head_length=1)
	plt.arrow(0,0,10*v[0,1],10*v[1,1],color='r',linewidth=2,head_width=1,head_length=1)
	plt.text(12*v[0,0],10*v[1,0],r'PC1, $\vec{v_1}$ =  %.2f $\vec{x_1}$ + %.2f $\vec{x_2}$' % (v[0,0],v[1,0]), fontsize=15)
	plt.text(26*v[0,1],8*v[1,1],r'PC2, $\vec{v_2}$ =  %.2f $\vec{x_&}$ + %.2f $\vec{x_2}$' % (v[0,1],v[1,1]), fontsize=15)
	
def convert_ranking(ranking,reverse=False):
	return sorted(range(len(ranking)), key=lambda k: ranking[k], reverse=reverse)

def plot_coefs(lambdas,coefs):
	ax = plt.gca()
	ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
	ax.plot(lambdas, coefs)
	lineObjects = ax.plot(lambdas, coefs)
	ax.set_xscale('log')	
	plt.legend(iter(lineObjects),[r"$\theta_{" + s + "}$" for s in map(str,range(1,13))])	
	plt.xlabel(r'$\lambda$')
	plt.ylabel(r'$\theta$')

dictionary = {"A": "adenine", "C": "cytosine", "G": "guanine", "T": "thymine","U":"unknown"}

