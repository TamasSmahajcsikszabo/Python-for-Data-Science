import tensorflow as tf

# graph
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y+y+2
tf.V
# a session
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

# a better way (variant)
with tf.Session() as sess:
	x.initializer.run()
	y.initializer.run()
	result = f.eval() 
# automatically closes

# an even better way
init = tf.global_variables_initializer() # prepares the init 

with tf.Session() as sess:
	init.run() # runs alls variables
	result = f.eval()
print(result)

# an alternative is an interactive session
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close() # has to be closed manually

# default sessions - managing multiple sessions
graph = tf.Graph()
with graph.as_default():
	x2 = tf.Variable(3)
# graph is no longer the default session as the block ends

# there is no code reuse in running a session, 'x' and 'W' are evaluated twice

w = tf.constant(5)
x = w + 2
y = x + 3
z = y + 4

with tf.Session() as sess:
	print(y.eval())
	print(z.eval())

# operations = ops
# tensors = inputs and outputs - they are multidimensional arrays, Numpy ndarray formats


### LINEAR REGRESSION ###
import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

## the normal equation
x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='x')
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
xT = tf.transpose(x)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xT, x)),xT),y)

with tf.Session() as sess:
	theta_value = theta.eval()

## Batch Gradient Descent
# IMPORTANT: normalize first
import tensorflow
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housing.data = scaler.fit_transform(housing.data)
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

n_epochs = 1000
learning_rate = 0.01
x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='x')
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1 ], -1.0, 1.0), name='theta')
y_pred = tf.matmul(x, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = 2/m * tf.matmul(tf.transpose(x), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print("Epoch", epoch, "MSE = ", mse.eval())
		sess.run(training_op)
	best_theta = theta.eval()

## autodiff

n_epochs = 1000
learning_rate = 0.01
x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='x')
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1 ], -1.0, 1.0), name='theta')
y_pred = tf.matmul(x, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = tf.gradients(mse, theta)[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print("Epoch", epoch, "MSE = ", mse.eval())
		sess.run(training_op)
	best_theta = theta.eval()

# using optimizers

n_epochs = 1000
learning_rate = 0.01
x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='x')
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1 ], -1.0, 1.0), name='theta')
y_pred = tf.matmul(x, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = tf.gradients(mse, theta)[0]
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print("Epoch", epoch, "MSE = ", mse.eval())
		sess.run(training_op)
	best_theta = theta.eval()

# changing the code to run minibatch
# with placeholder nodes, we can replace the original X and Y with iterated minibatches
# placeholders are special nodes that without computations, just return the data
# so they pass the data to TF during training

# a little demo
A = tf.placeholder(tf.float32, shape=(None, 3)) # without shape specification, it returns "any size"
B = A + 3
with tf.Session() as sess:
	B_val_1 = B.eval(feed_dict={A:[[1,2,3]]})
	B_val_2 = B.eval(feed_dict={A:[[1,2,3], [7,8,9]]})
	## feed_dict is fed in
	## eval() specifies the value of A

print(B_val_1)
print(B_val_2)

# 1. CONSTRUCION PHASE
# # modifying the original data so x and y are placeholders now:
x = tf.placeholder(tf.float32, shape=(None, n + 1), name = "x")
y = tf.placeholder(tf.float32, shape=(None, 1), name = "y")

# then the batch size has to be defined and the total number of batches computed
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

# 2. EXECUTION PHASE
# feed in the new values via the feed_dict parameter
def fetch_batch(epoch, batch_index, batch_size):
	np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
	indices = np.random.randint(m, size=batch_size)  # not shown
	x_batch = housing_data_plus_bias[indices]  # not shown
	y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
	return x_batch, y_batch

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
	best_theta = theta.eval()

## Saving and restoring models
# 1. define a saver node at the end of the construction phase after the variables have been created
# 2. in the execution phase call the save() method with the session passed into it and also the path
# to the checkpoint file
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
			save_path = saver.save(sess,"/home/tamassmahajcsikszabo/OneDrive/python_code/ML/checkpoint_model.ckpt")
	best_theta = theta.eval()
	save_path = saver.save(sess, "/home/tamassmahajcsikszabo/OneDrive/python_code/ML/final_minibatch_model.ckpt")

# to restore a model
# 1. declare a saver as before in the Construction Phase
# 2. call the saver's restore() method with session call and path to the stored model
with tf.Session() as sess:
	saver.restore(sess, "/home/tamassmahajcsikszabo/OneDrive/python_code/ML/final_minibatch_model.ckpt")
	# instead of an init node

# to constrain the saver to save only specified variables
saver = tf.train.Saver({"weights" : theta}) # this will save only theta under the name weights

# save() saves the graph, too in a "meta" file which can be restored, too
# i.e. not just the variables, but the whole graph
saver = tf.train.import_meta_graph("/home/tamassmahajcsikszabo/OneDrive/python_code/ML/final_minibatch_model.ckpt.meta")
# then the saver can be called as above to restore the variables, too

# Visualising the graph and training curves using Tensorboard
from datetime import datetime
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()
# creating a timestamp for a log directory TB can use
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)


n_epochs = 1000
learning_rate = 0.01
x = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(x, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE', mse) # writes into event files
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			if batch_index % 10 == 0:
				summary_str = mse_summary.eval(feed_dict={x : x_batch, y : y_batch})
				step = epoch * n_batches + batch_index
				file_writer.add_summary(summary_str, step)
			sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
	file_writer.close()

## name scopes can be created to group related nodes
# mse and error are now under the name scope "loss"
# from now on when printed each op within the scope is prefixed with "loss/"
# in tensorboard, now they belong in the loss bubble
import tensorflow as tf
from datetime import datetime
import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
def fetch_batch(epoch, batch_index, batch_size):
	np.random.seed(epoch * n_batches + batch_index)
	indices = np.random.randint(m, size=batch_size)
	x_batch = housing_data_plus_bias[indices]
	y_batch = housing.target.reshape(-1, 1)[indices]
	return x_batch, y_batch

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01
x = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(x, theta, name="predictions")
with tf.name_scope("loss") as scope:
	error = y_pred - y
	mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE', mse) # writes into event files
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			if batch_index % 10 == 0:
				summary_str = mse_summary.eval(feed_dict={x : x_batch, y : y_batch})
				step = epoch * n_batches + batch_index
				file_writer.add_summary(summary_str, step)
			sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
	file_writer.close()

## modularity
# rectified units (relu)
# they compute the linear function of the inputs and outputs the result if positive, or 0 otherwise

# a manual, hard-to-maintain example:
n_features = 3
x = tf.placeholder(tf.float32, shape=(None, n_features), name="x")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")
z1 = tf.add(tf.matmul(x, w1), b1, name="z1")
z2 = tf.add(tf.matmul(x, w1), b1, name="z2")

relu1 = tf.maximum(z1, 0, name="relu1")
relu2 = tf.maximum(z1, 0, name="relu2")

## INSTEAD:
# building relu with a function
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
def relu(x):
	with tf.name_scope("relu"):
		w_shape = (int(x.get_shape()[1]),1)
		w = tf.Variable(tf.random_normal(w_shape), name='weights')
		b = tf.Variable(0.0, name="bias")
		z = tf.add(tf.matmul(x, w), b, name="z")
		return tf.maximum(z,0,name="relu")

n_features = 3
x = tf.placeholder(tf.float32, shape=(None, n_features), name="x")
relus = [relu(x) for i in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()

## sharing variables
def relu(x, thershold):
	with tf.name_scope("relu"):
		w_shape = (int(x.get_shape()[1]),1)
		w = tf.Variable(tf.random_normal(w_shape), name='weights')
		b = tf.Variable(0.0, name="bias")
		z = tf.add(tf.matmul(x, w), b, name="z")
		return tf.maximum(z,threshold,name="max")

threshold = tf.Variable(0.0, name="threshold")
x = tf.placeholder(tf.float32, shape=(None, n_features), name="x")
relus = [relu(x, threshold) for i in range(5)]
output = tf.add_n(relus, name="output")

# but if the number of shared parameters increases, another solution is needed
# 1. create a dictionary with all the parameters in the model
# 2. create a class for the same purpose
# 3. set the shared variable as an attribute of the relu() function:

def relu(x):
	with tf.name_scope("relu"):
		relu.threshold = tf.Variable(0.0, name="threshold")
		w_shape = (int(x.get_shape()[1]),1)
		w = tf.Variable(tf.random_normal(w_shape), name='weights')
		b = tf.Variable(0.0, name="bias")
		z = tf.add(tf.matmul(x, w), b, name="z")
		return tf.maximum(z,relu.threshold,name="max")

x = tf.placeholder(tf.float32, shape=(None, n_features), name="x")
relus = [relu(x) for i in range(5)]
output = tf.add_n(relus, name="output")

# 4. tensorflow offers the option of get_variable()
# it creates (if doesn't exist) or reuses (if exists) the shared variable
with tf.variable_scope("relu"):
	threshold = tf.get_variable("threshold", shape=(),
							 initializer=tf.constant_initializer(0.0))
# reuse is controlled by the parameter within get_variable()
with tf.variable_scope("relu", reuse=True):
	threshold = tf.get_variable("threshold") # in this case, there is no need for the shape

# the modified function
def relu(x):
	with tf.variable_scope("relu", reuse=True):
		threshold = tf.get_variable("threshold")
		w_shape = (int(x.get_shape()[1]),1)
		w = tf.Variable(tf.random_normal(w_shape), name='weights')
		b = tf.Variable(0.0, name="bias")
		z = tf.add(tf.matmul(x, w), b, name="z")
		return tf.maximum(z,threshold,name="max")

x = tf.placeholder(tf.float32, shape=(None, n_features), name="x")
with tf.variable_scope("relu"):
	threshold = tf.get_variable("threshold", shape=(),
							 initializer=tf.constant_initializer(0.0))
relus = [relu(x) for i in range(5)]
output = tf.add_n(relus, name="output")

# the final version where threshold is created within the relu() function upon the first call,
# and reuses it in subsequent runs
reset_graph()

def relu(X):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
    w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
    b = tf.Variable(0.0, name="bias")                           # not shown
    z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
    return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")