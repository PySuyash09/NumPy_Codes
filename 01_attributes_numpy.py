import numpy as np

arr_1D = np.array([[1,2,3]])
print(arr_1D)

# ============= .ndim ============== #
'''Returns the number of dimensions of array'''
print(arr_1D.ndim)

# ============= .size ============== #
'''Returns the number of total elements of array'''
print(arr_1D.size)

# ============= .dtype ============== #
'''Returns the data type of elements of array'''
print(arr_1D.dtype)

# ============= .shape ============== #
'''Returns the shape (rows and elements in rows(columns)) of array'''
print(arr_1D.shape)

# ============= .itemsize ============== #
'''Returns the number of bytes every single element of array is used to store in memory (int64 represents data in 64 bits so it stores 64/8 = 8 bytes in memory)'''
print(arr_1D.itemsize)

# ============= .nbytes ============== #
'''Returns total number of bytes consumed by the array's data in memory. (arr.size * arr.itemsize)'''
print(arr_1D.nbytes)

# ============= .strides ============== #
'''The strides attribute tells how many bytes we need to skip in memory to move to the next element'''
print(arr_1D.strides)

# ============= .T ============== #
'''Transpose of array'''
print(arr_1D.T)

# ============= .flags ============== #
'''Gives information about memory layout of array'''
'''
C_CONTIGUOUS : Returns True if array elements are stored in rows in  continous block of memory. The elements are ordered such that the column index are changed so quickly. It is row major. ex -> [[1,2,3], [4,5,6]] ; Memory -> [1,2,3,4,5,6]

F_CONTIGUOUS : (Fortran-style/column-Major) The elements of array are stored in the memory in column order .
ex -> [[1,2,3],[4,5,6]] ; Memory -> [1,4,2,5,3,6]

OWNDATA : Returns True if the data belongs to the array i.e., array is responsible to manage the memory of data, Returns False if the array is view of any other array i.e., it does not own's it data and represent data owned by any other array.

WRITEABLE : Returns True if you can freely change the values of the array elements. Returns False if any attempt to modify an element will result in a ValueError: assignment destination is read-only or similar error.

ALIGNED : ALIGNED is a Boolean flag that is True if and only if the array's data buffer is properly aligned in memory, according to the requirements of the CPU and the array's data type (itemsize). For a standard 64-bit integer (itemsize=8), alignment means the data must start at an address that is divisible by 8 (e.g., address 1000, 1008, 1016, etc.).

WRITEBACKIFCOPY : WRITEBACKIFCOPY is a Boolean flag that is True when the current array is a temporary copy of another array's data, and this temporary copy needs to be written back to the original array's memory location when the temporary array is destroyed (i.e., when its reference count drops to zero).
'''
print(arr_1D.flags)

# ============= .flat ============== #
'''The ndarray.flat attribute in NumPy is an iterator object that allows you to treat a multi-dimensional array as a single, one-dimensional sequence, regardless of its actual shape. arr.flat does not return a new, flattened array (like arr.flatten() or arr.ravel()). Instead, it returns a special numpy.flatiter object, which is an efficient iterator.'''
for x in arr_1D.flat:
    print(x, end="\t")
print("\n")

# ============= .real, .imag ============== #
'''Access real and imaginary parts (complex arrays)'''
c = np.array([1+3j, 4-6j])
print(c.real)
print(c.imag)

# ============= .base ============== #
'''If the array is a view: The base attribute will contain a reference to the base array (the array whose data buffer is being shared). If the array owns its own data: The base attribute will be None'''
print(arr_1D.base)

# ============= .data ============== #
'''A buffer object pointing to the start of the array's memory block.'''
print(arr_1D.data) 