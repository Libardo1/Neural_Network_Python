import numpy as np

#sigmoid function
def nonlin(x, deriv=False):
	if deriv==True:
		return x*(1-x)
	return 1/(1+np.exp(-x))

# input dataset
X = np.array([[0,0,1],
				[0,1,1],
				[1,0,1],
				[1,1,1]])

#output dataset
y = np.array([[0,0,1,1]]).T

# set seed
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):

	#forward propagation
	l0 = X
	l1 = nonlin(l0.dot(syn0))

	# Error
	l1_error = y - l1

	# gradient multiply errors by the slope of sigmoid at value l1 (= derivate)
	l1_delta = l1_error * nonlin(l1, True)

	# update weights
	syn0 += l0.T.dot(l1_delta)

print "Output after training:"
print l1