 ### Lagrange Interpolant ###
import numpy as np

# Program suite that constructs an interpolating polynomial using the Lagrange method with barycentric weights


# phi(x) is the product of all the x-xi differences for i=0,1,..,n
# points is the vector/list of given x coordinates of the function we want to interpolate
# x is the scalar input value such that we have x - xi
def compute_phi(points,x):
    term = 1
    for k in range(len(points)):
        term *= (x-points[k])
    return term

# Computing the barycentric weights wj's such that wj = 1/w'(xj) where each entry in w'(xj) is the product of all the xj - xi differences for i=0,1,..,n, i != j
# x is the vector/list of x coordinates
# points are all the x_coordinate values available

def compute_w(points):
    w_prime=[]
    for i in range(len(points)):
        term = 1
        for k in range(len(points)):
            if (i!= k):
                term *= (points[i]-points[k])
        w_prime.append(term)
    return 1/(np.asarray(w_prime))

# The difference between x and each xj
def x_minus_x_j(points,x):
    mySum = []
    for i in range(len(points)):
        mySum.append( x - points[i])
    return np.array(mySum)

# constructing the interpolants using the barycentric coefficients w such that  [ phi(x) * sum of f(xk)*wk/(x-xk) for k=0,1,..,n] =p(x)
# This is the main function the problem asks for
def barycentric_weights_construction(x_coordinates,x):
    if (x in np.asarray(x_coordinates)):
        return (np.asarray(x_coordinates)==x) + np.array([0,0,0])
    xdiff = x_minus_x_j(x_coordinates,x)
    w = compute_w(x_coordinates)
    phi = compute_phi(x_coordinates,x)

    return phi*(w/xdiff)
    
# computing the lagrange interpolation estimate of the function at x
# inputs:
# x_coords: the x-coordinates at which we are interpolating the polynomial
# f_values: the function evaluation at the corresponding x-coordinates
# x: the x value we want to extrapolate(ie we want to find f(x), the evaluation of the polynomial @x)

def evaluate_lagrange(x_coords,f_values,x):
    w=np.asarray(barycentric_weights_construction(x_coords,x))
    print('The barycentric weights are: ' + repr(w))
    print('The approximation of the function at x=' + repr(x) + ' is ' + repr(np.dot(w,f_values)))
    return w,np.dot(w,f_values)

### Testing ###

x_coords=[0,1,3]
x = 1

f_values = np.array([4,10,0])

evaluate_lagrange(x_coords,f_values,x)


