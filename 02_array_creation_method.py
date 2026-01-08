import numpy as np


#================== np.array() ==========================#
'''Create an array from a list, tuple, or nested list.'''
'''Syntax: numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0, *, like=None)'''
'''like parameter : The function attempts to infer properties (like data type, memory layout) from the like argument to enable downstream operations (like ufuncs) that require a specific array context. This is a recent parameter (NumPy 1.20+) and is primarily used internally or by advanced libraries.'''
array = np.array([1,2,3,4])
print(array)

#================== np.zeros() ==========================#
'''Its purpose is to create a new NumPy array of a specified shape and data type, filled entirely with zeros.'''
'''Syntax: numpy.zeros(shape, dtype=float, order='C', *, like=None)'''
zeros = np.zeros(shape=(2,3), dtype=np.int64, order='C')
print(zeros)

#================== np.ones() ==========================#
'''np.ones() is a NumPy function used to create a new array filled entirely with the value 1.'''
'''Syntax: np.ones(shape, dtype=None, order='C')'''
ones = np.ones(shape=(2,3), dtype=np.int64, order='C')
print(ones)

#================== np.empty() ==========================#
'''np.empty() is a NumPy function used to create a new array without initializing its values (Garbage values).'''
'''Syntax: np.empty(shape, dtype=None, order='C')'''
empty = np.empty(shape=(2,3), dtype=np.int64, order='C')
print(empty)

#================== np.arange() ==========================#
'''np.arange() is a NumPy function used to create an array with evenly spaced values within a given range, similar to Python’s built-in range(), but it returns a NumPy array.'''
'''Syntax: np.arange(start, stop, step, dtype=None)'''
arange = np.arange(start=1, stop=10, step=2, dtype=np.int64)
print(arange)

#================== np.linspace() ==========================#
'''np.linspace() is a NumPy function used to generate a specified number of evenly spaced values between two numbers.'''
'''Syntax: np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)'''
linspace, step = np.linspace(start=0, stop=1, num=5, endpoint=False, retstep=True)
print(linspace)
print(step)

#================== np.asarray() ==========================#
'''The np.asarray() method in NumPy is very similar to np.array(), but its primary focus is on ensuring the output is an ndarray without unnecessarily copying data.
np.asarray() will create a view (i.e., return the original array object) only if the input array-like object (a) meets two conditions:
1. It is already an np.ndarray.
2. It matches the requested dtype and order (memory layout).'''
'''Syntax: numpy.asarray(a, dtype=None, order=None, *, like=None)'''

array = np.array([1,2,3], order='F', dtype=np.int64)
asarray = np.asarray(array, order='C', dtype=np.float64)
print(array)
print(asarray)
print(array.data)
print(asarray.data)

#================== np.eye() ==========================#
'''The np.eye() method in NumPy is a specialized function used to create a 2-dimensional (2D) array (a matrix) with ones on the k-th diagonal and zeros everywhere else.'''
'''Syntax: numpy.eye(N, M=None, k=0, dtype=float, order='C', *, like=None)'''
'''
N : Required. The number of rows in the output array.
M : Optional. The number of columns in the output array. If None, it defaults to N.
K : Optional. The index of the diagonal where ones are placed. Defaults to 0 (the main diagonal).
dtype : The data type of the array elements. Defaults to float64.'''
eye = np.eye(4, 4, dtype=np.int64, k=0)
print(eye)

#================== np.identity() ==========================#
'''The np.identity() method in NumPy is a specialized function used to create a square identity matrix.It is a more specific version of np.eye() where the number of rows equals the number of columns, and the ones are always placed strictly on the main diagonal (k=0).'''
'''Syntax: numpy.identity(n, dtype=None, *, like=None)'''
identity = np.identity(5)
print(identity)

#================== np.diag() ==========================#
'''The np.diag() method in NumPy is a versatile function that serves two primary, inverse purposes related to array diagonals: 
Extracting a specified diagonal from a 2D array.
Creating a 2D square matrix with a given 1D array as its diagonal.'''
'''Syntax: numpy.diag(v, k=0)
v : The input array. This determines whether the function extracts or constructs.
k : The index of the diagonal to use. Defaults to $0$ (the main diagonal).'''
# Creating a Diagonal Matrix (Input is 1D)
vector = np.array([1,2,3])
diag = np.diag(vector, k=0)
print(diag)
# Extracting a Diagonal (Input is 2D)
matrix = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
main_diag = np.diag(matrix, k=0)
print(main_diag)

#================== np.full() ==========================#
'''The np.full() method in NumPy is a very useful array creation function that allows you to create a new array of a specified shape and fill it entirely with a single, constant value.'''
'''Syntax : numpy.full(shape, fill_value, dtype=None, order='C', *, like=None)'''
full = np.full((4,4), 7, dtype=np.int32, order='C')
print(full)

#================== np.zeros_like(), np.ones_like(), np.empty_like() ==========================#
'''The NumPy functions np.zeros_like(), np.ones_like(), and np.empty_like() are collectively known as "like" functions. Their purpose is to create a new array that has the same properties (shape and data type) as a pre-existing array, but filled with a specific initial value.'''
'''Syntax : numpy.function_name(a, dtype=None, order='K', subok=True, shape=None, *, like=None)'''
prototype = np.array([[1,2,3],[4,5,6]])
zero_like = np.zeros_like(prototype)
print(zero_like)

#================== np.logspace() ==========================#
'''np.logspace() is a NumPy function used to create an array of numbers that are evenly spaced on a log scale. While np.linspace() creates numbers with equal differences between them (arithmetic progression), np.logspace() creates numbers where each element is a constant multiple of the previous one (geometric progression).'''
'''Syntax : numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)'''
'''start_value = base ^ start, end_value = base ^ end'''
log_arr = np.logspace(start=1, stop=10, num=5, dtype=np.int64, endpoint=True, base=2)
print(log_arr)

#================== np.geomspace() ==========================#
'''np.geomspace() is very similar to np.logspace(), but with one major practical difference: it uses the actual start and stop values rather than their exponents.It creates a sequence of numbers that are evenly spaced on a logarithmic scale, forming a geometric progression (where each number is the previous number multiplied by a constant).'''
'''numpy.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)'''
'''start_value = start, end_value = end'''
geomspace = np.geomspace(start=1, stop=1000, num=4, endpoint=True)
print(geomspace)


#================ Random Arrays ================================#
#================== np.random() ==========================#
'''np.random.rand() is a legacy NumPy function used to create an array of a given shape and populate it with random samples from a Uniform Distribution over the interval [0, 1]."Uniform Distribution" means that every number between 0 and 1 has an equal chance of being selected. Unlike most NumPy functions, rand does not take a tuple for the shape. You simply pass the dimensions as individual arguments.'''

# Older method
val = np.random.rand()  # Generates a sinngle value
print(val)

val = np.random.rand(3)     # Generates 1D array with three values
print(val)

val = np.random.rand(3,3)   # Generates 2D array with 3 rows and 3 column
print(val)

val = np.random.randint(1,10,5) # random.randint(start, stop, valuesCount) -> Generates 1D array
print(val)

val = np.random.randint(1,10,(2,3)) # random.randint(start, stop, valuesCount) -> Generates 2D array
print(val)

# Modern Method 
'''To use the new system, you first initialize a Generator object. This object acts like a "toolbox" for all your random needs.'''
rng = np.random.default_rng()   # Initilizing generator object.
floats = rng.random((3,2))  # (3,2) -> (rows, columns)
print("\n")
print(floats)

integers = rng.integers(1, 10, size=(3,3)) # 10 is exclusive
print("\n")
print(integers)

choice = rng.choice([x for x in range(1,30)], size=(4,4))   # Generates a given size array by randomly choosing the values from sequence.
print(choice)

r = np.random.randn(2,3) # np.random.randn() is a NumPy function used to create an array of a given shape populated with random samples from the Standard Normal (Gaussian) Distribution.
print(r)


#================================ np.fromiter ==========================================#
'''While np.array() wants to see the entire "finished" collection at once, np.fromiter() acts like a conveyor belt. It takes a single value, places it in the array, reaches back for the next single value, and repeats until the stream is empty.'''
'''Syntax : np.fromiter(sensor_stream(num_elements), dtype=float, count=num_elements)'''
'''count=num_elements: This is optional but highly recommended. It tells NumPy exactly how much memory to set aside at the start. If you don't provide count, NumPy has to keep resizing the array as more data comes in, which is much slower.'''
def stream_generator(total_points):
    '''Simulates a stream of 1D data points'''
    for i in range(total_points):
        value = i*1.23
        yield value     # yield : While a normal function uses return to send back a result and then terminates, a  function with yield sends back a value and pauses its execution, remembering exactly where it left off.

num_elements = 10
arr = np.fromiter(stream_generator(num_elements), dtype=float, count=num_elements)
print("\n")
print(arr)

#============================ np.fromfunction() ====================================#
'''np.fromfunction() is a unique NumPy tool that creates an array by executing a function over every coordinate in the grid. Instead of filling an array with existing data, you provide a rule (a function), and NumPy calculates the value of each element based on its index (row, column, etc.).'''
'''Syntax : numpy.fromfunction(function, shape, **kwargs, dtype(optional))'''
arr_diag = np.fromfunction(lambda i,j : i==j, (3,3)).astype(np.int64)   # astype() method is used to convert data from one type to another.
print(arr_diag)

#============================ np.tile() ====================================#
'''np.tile() is a NumPy function used to construct an array by repeating another array a specific number of times.
Think of it like laying down floor tiles. You have one single tile (your input array), and you "tile" it across a floor to create a larger pattern.'''
'''Syntax : np.tile(A, reps)'''
print("\n")
arr = np.array([1,2,3])
tile_arr = np.tile(arr, reps=3)
print(tile_arr)
print("\n")
arr = np.array([1,2])
tile_arr = np.tile(arr, reps=(3,3))
print(tile_arr)

#============================ np.repeat() ====================================#
'''np.repeat() is a NumPy function used to repeat the individual elements of an array. While np.tile() repeats the entire block of an array, np.repeat() stays on each element and replicates it a specified number of times before moving to the next one.'''
'''Syntax : np.repeat(a, repeats, axis=None)'''
'''a: The input array.
repeats: How many times to repeat each element (can be a single integer or an array of integers).
axis: The axis along which to repeat. If None (default), the array is flattened'''
arr = np.array([1, 2, 3])
repeated = np.repeat(arr, 3)
print(repeated)
arr = np.array([1, 2])
custom_repeat = np.repeat(arr, [2, 4])
print(custom_repeat)