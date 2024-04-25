import numpy as np

import pandas as pd

#1. Create a NumPy array 'arr' of integers from 0 to 5 and print its data type.

import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5])

print(arr.dtype)


#2.  Given a NumPy array 'arr', check if its data type is float64

arr = np.array([1.5, 2.6, 3.7])

print(arr.dtype)

#3. Create a NumPy array 'arr' with a data type of complex128 containing three complex numbers.
 

arr = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)

print(arr)

#4. Convert an existing NumPy array 'arr' of integers to float32 data type.


arr = np.array([1, 2, 3])
arr_float32 = arr.astype(np.float32)
print(arr_float32)

#5. Given a NumPy array 'arr' with float64 data type, convert it to float32 to reduce decimal precision.


arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
arr_float32 = arr.astype(np.float32)
print(arr_float32)



# 6. Write a function array_attributes that takes a NumPy array as input and returns its shape, size, and data type.
def array_attributes(arr):
    return arr.shape, arr.size, arr.dtype

# Example
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(array_attributes(arr))  # Output: ((2, 3), 6, dtype('int64'))

# 7. Create a function array_dimension that takes a NumPy array as input and returns its dimensionality.
def array_dimension(arr):
    return arr.ndim

# Example
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(array_dimension(arr))  # Output: 2

# 8. Design a function item_size_info that takes a NumPy array as input and returns the item size and the total size in bytes.
def item_size_info(arr):
    return arr.itemsize, arr.nbytes

# Example
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(item_size_info(arr))  # Output: (8, 48)

# 10. Design a function shape_stride_relationship that takes a NumPy array as input and returns the shape and strides of the array.
def shape_stride_relationship(arr):
    return arr.shape, arr.strides

# Example
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(shape_stride_relationship(arr))  # Output: ((2, 3), (24, 8))

# 11. Create a function create_zeros_array that takes an integer n as input and returns a NumPy array of zeros with n elements.
def create_zeros_array(n):
    return np.zeros(n)

# Example
n = 5
print(create_zeros_array(n))  # Output: [0. 0. 0. 0. 0.]

# 12. Write a function create_ones_matrix that takes integers rows and cols as inputs and generates a 2D NumPy array filled with ones of size rows x cols.
def create_ones_matrix(rows, cols):
    return np.ones((rows, cols))

# Example
rows, cols = 2, 3
print(create_ones_matrix(rows, cols))  # Output: [[1. 1. 1.], [1. 1. 1.]]

# 13. Write a function generate_range_array that takes three integers start, stop, and step as arguments and creates a NumPy array with a range starting from start, ending at stop (exclusive), and with the specified step.
def generate_range_array(start, stop, step):
    return np.arange(start, stop, step)

# Example
start, stop, step = 0, 5, 1
print(generate_range_array(start, stop, step))  # Output: [0 1 2 3 4]

# 14. Design a function generate_linear_space that takes two floats start, stop, and an integer num as arguments and generates a NumPy array with num equally spaced values between start and stop (inclusive).
def generate_linear_space(start, stop, num):
    return np.linspace(start, stop, num)

# Example
start, stop, num = 0, 1, 5
print(generate_linear_space(start, stop, num))  # Output: [0.   0.25 0.5  0.75 1.  ]

# 15. Create a function create_identity_matrix that takes an integer n as input and generates a square identity matrix of size n x n using numpy.eye.
def create_identity_matrix(n):
    return np.eye(n)

# Example
n = 3
print(create_identity_matrix(n))  # Output: [[1. 0. 0.], [0. 1. 0.], [0. 0. 1.]]

# 16. Write a function that takes a Python list and converts it into a NumPy array.
def convert_to_numpy_array(lst):
    return np.array(lst)

# Example
lst = [1, 2, 3, 4, 5]
print(convert_to_numpy_array(lst))  # Output: [1 2 3 4 5]

# 17. Create a NumPy array and demonstrate the use of numpy.view to create a new array object with the same data.
arr = np.array([1, 2, 3, 4, 5])
arr_view = arr.view()

# Example
print("Original array:", arr)  # Output: [1 2 3 4 5]
print("Array view:", arr_view)  # Output: [1 2 3 4 5]


# 18. Write a function that takes two NumPy arrays and concatenates them along a specified axis.
def concatenate_arrays(arr1, arr2, axis=0):
    return np.concatenate((arr1, arr2), axis=axis)

# Example
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])
print(concatenate_arrays(arr1, arr2, axis=0))  # Output: [[1 2] [3 4] [5 6]]

# 19. Create two NumPy arrays with different shapes and concatenate them horizontally using `numpy.concatenate`.
arr1 = np.array([[1, 2, 3]])
arr2 = np.array([[4], [5], [6]])
print(np.concatenate((arr1, arr2), axis=1))  # Output: [[1 2 3 4] [5 6 0 0] [7 8 0 0]]

# 20. Write a function that vertically stacks multiple NumPy arrays given as a list.
def stack_arrays_vertically(arrays):
    return np.vstack(arrays)

# Example
arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
print(stack_arrays_vertically(arrays))  # Output: [[1 2 3] [4 5 6]]

# 21. Write a Python function using NumPy to create an array of integers within a specified range (inclusive) with a given step size.
def create_range_array(start, stop, step):
    return np.arange(start, stop+1, step)

# Example
start, stop, step = 1, 10, 2
print(create_range_array(start, stop, step))  # Output: [1 3 5 7 9]

# 22. Write a Python function using NumPy to generate an array of 10 equally spaced values between 0 and 1 (inclusive).
def generate_equally_spaced_array(num):
    return np.linspace(0, 1, num)

# Example
num = 10
print(generate_equally_spaced_array(num))  # Output: [0.         0.11111111 0.22222222 0.33333333 0.44444444 0.55555556 0.66666667 0.77777778 0.88888889 1.        ]

# 23. Write a Python function using NumPy to create an array of 5 logarithmically spaced values between 1 and 1000 (inclusive).
def generate_log_spaced_array(num):
    return np.logspace(0, 3, num)

# Example
num = 5
print(generate_log_spaced_array(num))  # Output: [   1.   10.  100. 1000. 10000.]

# 24. Create a Pandas DataFrame using a NumPy array that contains 5 rows and 3 columns, where the values are random integers between 1 and 100.
data = np.random.randint(1, 101, (5, 3))
df = pd.DataFrame(data)

# Example
print(df)

# 25. Write a function that takes a Pandas DataFrame and replaces all negative values in a specific column with zeros. Use NumPy operations within the Pandas DataFrame.
def replace_negative_with_zeros(df, column):
    df[column] = np.where(df[column] < 0, 0, df[column])
    return df

# Example
data = {'A': [1, -2, 3, -4, 5], 'B': [-1, 2, -3, 4, -5]}
df = pd.DataFrame(data)
print(replace_negative_with_zeros(df, 'B'))

# 26. Access the 3rd element from the given NumPy array.
arr = np.array([1, 2, 3, 4, 5])
print(arr[2])  # Output: 3

# 27. Retrieve the element at index (1, 2) from the 2D NumPy array.
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[1, 2])  # Output: 6

# 28. Using boolean indexing, extract elements greater than 5 from the given NumPy array.
arr = np.array([1, 6, 2, 7, 3, 8, 4, 9, 5])
print(arr[arr > 5])  # Output: [6 7 8 9]

# 29. Perform basic slicing to extract elements from index 2 to 5 (inclusive) from the given NumPy array.
arr = np.array([10, 20, 30, 40, 50, 60, 70])
print(arr[2:6])  # Output: [30 40 50 60]

# 30. Slice the 2D NumPy array to extract the sub-array `[[2, 3], [5, 6]]` from the given array.
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[0:2, 1:])  # Output: [[2 3] [5 6]]


# 31. Write a NumPy function to extract elements in specific order from a given 2D array based on indices provided in another array.
def extract_elements(arr, indices):
    return arr.flatten()[indices]

# Example
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
indices = np.array([1, 5, 8])
print(extract_elements(arr, indices))  # Output: [2 5 8]

# 32. Create a NumPy function that filters elements greater than a threshold from a given 1D array using boolean indexing.
def filter_greater_than_threshold(arr, threshold):
    return arr[arr > threshold]

# Example
arr = np.array([1, 10, 2, 20, 3, 30])
threshold = 5
print(filter_greater_than_threshold(arr, threshold))  # Output: [10 20 30]

# 33. Develop a NumPy function that extracts specific elements from a 3D array using indices provided in three separate arrays for each dimension.
def extract_elements_3d(arr, indices_0, indices_1, indices_2):
    return arr[indices_0, indices_1, indices_2]

# Example
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
indices_0 = np.array([0, 1, 1])
indices_1 = np.array([0, 1, 0])
indices_2 = np.array([1, 0, 1])
print(extract_elements_3d(arr, indices_0, indices_1, indices_2))  # Output: [ 2  7 10]

# 34. Write a NumPy function that returns elements from an array where both two conditions are satisfied using boolean indexing.
def filter_elements_with_conditions(arr, condition1, condition2):
    return arr[np.logical_and(condition1, condition2)]

# Example
arr = np.array([1, 10, 2, 20, 3, 30])
condition1 = arr > 5
condition2 = arr % 2 == 0
print(filter_elements_with_conditions(arr, condition1, condition2))  # Output: [10 20 30]

# 35. Create a NumPy function that extracts elements from a 2D array using row and column indices provided in separate arrays.
def extract_elements_2d(arr, row_indices, col_indices):
    return arr[row_indices, col_indices]

# Example
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_indices = np.array([0, 1, 2])
col_indices = np.array([0, 1, 2])
print(extract_elements_2d(arr, row_indices, col_indices))  # Output: [1 5 9]

# 36. Given an array arr of shape (3, 3), add a scalar value of 5 to each element using NumPy broadcasting.
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = arr + 5
print(result)  # Output: [[ 6  7  8] [ 9 10 11] [12 13 14]]

# 37. Consider two arrays arr1 of shape (1, 3) and arr2 of shape (3, 4). Multiply each row of arr2 by the corresponding element in arr1 using NumPy broadcasting.
arr1 = np.array([[1, 2, 3]])
arr2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
result = arr1 * arr2
print(result)  # Output: [[ 1  4  9 16] [ 5 12 21 32] [ 9 20 33 48]]

# 38. Given a 1D array arr1 of shape (1, 4) and a 2D array arr2 of shape (4, 3), add arr1 to each row of arr2 using NumPy broadcasting.
arr1 = np.array([[1, 2, 3, 4]])
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
result = arr1 + arr2
print(result)  # Output: [[ 2  4  6] [ 5  7  9] [ 8 10 12] [11 13 15]]

# 39. Consider two arrays arr1 of shape (3, 1) and arr2 of shape (1, 3). Add these arrays using NumPy broadcasting.
arr1 = np.array([[1], [2], [3]])
arr2 = np.array([[4, 5, 6]])
result = arr1 + arr2
print(result)  # Output: [[5 6 7] [6 7 8] [7 8 9]]

# 40. Given arrays arr1 of shape (2, 3) and arr2 of shape (2, 2), perform multiplication using NumPy broadcasting. Handle the shape incompatibility.
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[1, 2], [3, 4]])
result = arr1 * arr2[:, :2]  # Handle shape incompatibility by slicing arr2
print(result)  # Output: [[ 1  4  6] [12 20 24]]

# 41. Calculate column wise mean for the given array:
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.mean(arr, axis=0)
print(result)  # Output: [2.5 3.5 4.5]

# 42. Find maximum value in each row of the given array:
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.max(arr, axis=1)
print(result)  # Output: [3 6]

# 43. For the given array, find indices of maximum value in each column.
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.argmax(arr, axis=0)
print(result)  # Output: [1 1 1]

# 44. For the given array, apply custom function to calculate moving sum along rows.
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = np.apply_along_axis(lambda x: np.convolve(x, np.ones(2), mode='valid'), axis=1, arr=arr)
print(result)  # Output: [[ 3.  5.] [ 9. 11.]]

# 45. In the given array, check if all elements in each column are even.
arr = np.array([[2, 4, 6], [3, 5, 7]])
result = np.all(arr % 2 == 0, axis=0)
print(result)  # Output: [False False False]

# 46. Given a NumPy array arr, reshape it into a matrix of dimensions `m` rows and `n` columns. Return the reshaped matrix.
def reshape_array(arr, m, n):
    return arr.reshape(m, n)

# Example
arr = np.array([1, 2, 3, 4, 5, 6])
m = 2
n = 3
print(reshape_array(arr, m, n))  # Output: [[1 2 3] [4 5 6]]

# 47. Create a function that takes a matrix as input and returns the flattened array.
def flatten_matrix(matrix):
    return matrix.flatten()

# Example
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(flatten_matrix(matrix))  # Output: [1 2 3 4 5 6]

# 48. Write a function that concatenates two given arrays along a specified axis.
def concatenate_arrays(arr1, arr2, axis=0):
    return np.concatenate((arr1, arr2), axis=axis)

# Example
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])
print(concatenate_arrays(arr1, arr2, axis=0))  # Output: [[1 2] [3 4] [5 6]]

# 49. Create a function that splits an array into multiple sub-arrays along a specified axis.
def split_array(arr, num_sections, axis=0):
    return np.split(arr, num_sections, axis=axis)

# Example
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(split_array(arr, num_sections=3, axis=0))  # Output: [array([[1, 2, 3]]), array([[4, 5, 6]]), array([[7, 8, 9]])]

# 50. Write a function that inserts and then deletes elements from a given array at specified indices.
def insert_and_delete_elements(arr, indices_insert, values_insert, indices_delete):
    arr = np.insert(arr, indices_insert, values_insert)
    arr = np.delete(arr, indices_delete)
    return arr

# Example
arr = np.array([1, 2, 3, 4, 5])
indices_insert = [1, 3]
values_insert = [10, 20]
indices_delete = [0, 4]
print(insert_and_delete_elements(arr, indices_insert, values_insert, indices_delete))  # Output: [ 2 10  3 20]

# 51. Create a NumPy array `arr1` with random integers and another array `arr2` with integers from 1 to 10. Perform element-wise addition between `arr1` and `arr2`.
arr1 = np.random.randint(1, 100, size=(3, 3))
arr2 = np.arange(1, 10).reshape(3, 3)
result = arr1 + arr2
print(result)

# 52. Generate a NumPy array `arr1` with sequential integers from 10 to 1 and another array `arr2` with integers from 1 to 10. Subtract `arr2` from `arr1` element-wise.
arr1 = np.arange(10, 0, -1)
arr2 = np.arange(1, 11)
result = arr1 - arr2
print(result)

# 53. Create a NumPy array `arr1` with random integers and another array `arr2` with integers from 1 to 5. Perform element-wise multiplication between `arr1` and `arr2`.
arr1 = np.random.randint(1, 100, size=(3, 3))
arr2 = np.arange(1, 6).reshape(3, 1)
result = arr1 * arr2
print(result)

# 54. Generate a NumPy array `arr1` with even integers from 2 to 10 and another array `arr2` with integers from 1 to 5. Perform element-wise division of `arr1` by `arr2`.
arr1 = np.arange(2, 11, 2)
arr2 = np.arange(1, 6)
result = arr1 / arr2
print(result)

# 55. Create a NumPy array `arr1` with integers from 1 to 5 and another array `arr2` with the same numbers reversed. Calculate the exponentiation of `arr1` raised to the power of `arr2` element-wise.
arr1 = np.arange(1, 6)
arr2 = np.flip(arr1)
result = arr1 ** arr2
print(result)

# 56. Write a function that counts the occurrences of a specific substring within a NumPy array of strings.
def count_substring_occurrences(arr, substring):
    return np.sum([substring in s for s in arr])

# Example
arr = np.array(['hello', 'world', 'hello', 'numpy', 'hello'])
substring = 'hello'
print(count_substring_occurrences(arr, substring))  # Output: 3

# 57. Write a function that extracts uppercase characters from a NumPy array of strings.
def extract_uppercase_characters(arr):
    return np.array([''.join([c for c in s if c.isupper()]) for s in arr])

# Example
arr = np.array(['Hello', 'World', 'OpenAI', 'GPT'])
print(extract_uppercase_characters(arr))  # Output: ['H' 'W' 'O' 'GPT']

# 58. Write a function that replaces occurrences of a substring in a NumPy array of strings with a new string.
def replace_substring(arr, old_substring, new_substring):
    return np.array([s.replace(old_substring, new_substring) for s in arr])

# Example
arr = np.array(['apple', 'banana', 'grape', 'pineapple'])
old_substring = 'apple'
new_substring = 'orange'
print(replace_substring(arr, old_substring, new_substring))

# 59. Write a function that concatenates strings in a NumPy array element-wise.
def concatenate_strings(arr):
    return np.array([''.join(s) for s in arr])

# Example
arr = np.array([['a', 'b'], ['c', 'd']])
print(concatenate_strings(arr))

# 60. Write a function that finds the length of the longest string in a NumPy array.
def longest_string_length(arr):
    return max([len(s) for s in arr])

# Example
arr = np.array(['apple', 'banana', 'grape', 'pineapple'])
print(longest_string_length(arr))

# 61. Create a dataset of 100 random integers between 1 and 1000. Compute the mean, median, variance, and standard deviation of the dataset using NumPy's functions.
dataset = np.random.randint(1, 1000, size=100)
mean = np.mean(dataset)
median = np.median(dataset)
variance = np.var(dataset)
std_dev = np.std(dataset)
print("Mean:", mean, "Median:", median, "Variance:", variance, "Standard Deviation:", std_dev)

# 62. Generate an array of 50 random numbers between 1 and 100. Find the 25th and 75th percentiles of the dataset.
data = np.random.randint(1, 100, size=50)
percentile_25th = np.percentile(data, 25)
percentile_75th = np.percentile(data, 75)
print("25th Percentile:", percentile_25th, "75th Percentile:", percentile_75th)

# 63. Create two arrays representing two sets of variables. Compute the correlation coefficient between these arrays using NumPy's `corrcoef` function.
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([5, 4, 3, 2, 1])
correlation_coefficient = np.corrcoef(array1, array2)[0, 1]
print("Correlation Coefficient:", correlation_coefficient)

# 64. Create two matrices and perform matrix multiplication using NumPy's `dot` function.
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
result = np.dot(matrix1, matrix2)
print(result)

# 65. Create an array of 50 integers between 10 and 1000. Calculate the 10th, 50th (median), and 90th percentiles along with the first and third quartiles.
data = np.random.randint(10, 1000, size=50)
percentiles = np.percentile(data, [10, 25, 50, 75, 90])
print("10th Percentile:", percentiles[0], "25th Percentile:", percentiles[1], "50th Percentile:", percentiles[2], "75th Percentile:", percentiles[3], "90th Percentile:", percentiles[4])

# 66. Create a NumPy array of integers and find the index of a specific element.
arr = np.array([1, 2, 3, 4, 5])
index = np.where(arr == 3)[0][0]
print("Index of 3:", index)

# 67. Generate a random NumPy array and sort it in ascending order.
arr = np.random.randint(1, 100, size=10)
sorted_arr = np.sort(arr)
print("Sorted Array:", sorted_arr)

# 68. Filter elements >20 in the given NumPy array.
arr = np.array([12, 25, 6, 42, 8, 30])
filtered_arr = arr[arr > 20]
print("Filtered Array:", filtered_arr)

# 69. Filter elements which are divisible by 3 from a given NumPy array.
arr = np.array([1, 5, 8, 12, 15])
filtered_arr = arr[arr % 3 == 0]
print("Filtered Array:", filtered_arr)

# 70. Filter elements which are ≥ 20 and ≤ 40 from a given NumPy array.
arr = np.array([10, 20, 30, 40, 50])
filtered_arr = arr[(arr >= 20) & (arr <= 40)]
print("Filtered Array:", filtered_arr)

# 71. For the given NumPy array, check its byte order using the `dtype` attribute byteorder.
arr = np.array([1, 2, 3])
byte_order = arr.dtype.byteorder
print("Byte Order:", byte_order)

# 72. For the given NumPy array, perform byte swapping in place using `byteswap()`.
arr = np.array([1, 2, 3], dtype=np.int32)
arr.byteswap(True)
print("Byte swapped array:", arr)

# 73. For the given NumPy array, swap its byte order without modifying the original array using `newbyteorder()`.
arr = np.array([1, 2, 3], dtype=np.int32)
new_arr = arr.newbyteorder()
print("Original array:", arr)
print("New byte order array:", new_arr)

# 74. For the given NumPy array and swap its byte order conditionally based on system endianness using `newbyteorder()`.
arr = np.array([1, 2, 3], dtype=np.int32)
new_arr = arr.newbyteorder('S')
print("Original array:", arr)
print("New byte order array:", new_arr)

# 75. For the given NumPy array, check if byte swapping is necessary for the current system using `dtype` attribute `byteorder`.
arr = np.array([1, 2, 3], dtype=np.int32)
byte_swap_needed = arr.dtype.byteorder not in ('=', '|')
print("Byte swapping necessary:", byte_swap_needed)

# 76. Create a NumPy array `arr1` with values from 1 to 10. Create a copy of `arr1` named `copy_arr` and modify an element in `copy_arr`. Check if modifying `copy_arr` affects `arr1`.
arr1 = np.arange(1, 11)
copy_arr = arr1.copy()
copy_arr[0] = 100
print("Original array:", arr1)
print("Modified copy array:", copy_arr)
print("Does modifying copy array affect original array?", np.array_equal(arr1, copy_arr))

# 77. Create a 2D NumPy array `matrix` of shape (3, 3) with random integers. Extract a slice `view_slice` from the matrix. Modify an element in `view_slice` and observe if it changes the original `matrix`.
matrix = np.random.randint(1, 10, (3, 3))
view_slice = matrix[:2, :2]
print("Original matrix:")
print(matrix)
print("View slice:")
print(view_slice)
view_slice[0, 0] = 100
print("Modified view slice:")
print(view_slice)
print("Changed original matrix:")
print(matrix)

# 78. Create a NumPy array `array_a` of shape (4, 3) with sequential integers from 1 to 12. Extract a slice `view_b` from `array_a` and broadcast the addition of 5 to view_b. Check if it alters the original `array_a`.
array_a = np.arange(1, 13).reshape(4, 3)
view_b = array_a[:2, :2]
view_b += 5
print("Original array_a:")
print(array_a)
print("Modified view_b:")
print(view_b)
print("Changed original array_a:")
print(array_a)

# 79. Create a NumPy array `orig_array` of shape (2, 4) with values from 1 to 8. Create a reshaped view `reshaped_view` of shape (4, 2) from orig_array. Modify an element in `reshaped_view` and check if it reflects changes in the original `orig_array`.
orig_array = np.arange(1, 9).reshape(2, 4)
reshaped_view = orig_array.reshape(4, 2)
reshaped_view[0, 0] = 100
print("Original array:")
print(orig_array)
print("Modified reshaped view:")
print(reshaped_view)
print("Changed original array:")
print(orig_array)

# 80. Create a NumPy array `data` of shape (3, 4) with random integers. Extract a copy `data_copy` of elements greater than 5. Modify an element in `data_copy` and verify if it affects the original `data`.
data = np.random.randint(1, 10, (3, 4))
data_copy = data[data > 5].copy()
data_copy[0] = 100
print("Original data:")
print(data)
print("Modified data_copy:")
print(data_copy)
print("Original data after modifying data_copy:")
print(data)

# 81. Create two matrices A and B of identical shape containing integers and perform addition and subtraction operations between them.
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
addition_result = A + B
subtraction_result = A - B
print("Addition result:")
print(addition_result)
print("Subtraction result:")
print(subtraction_result)

# 82. Generate two matrices `C` (3x2) and `D` (2x4) and perform matrix multiplication.
C = np.random.randint(1, 10, (3, 2))
D = np.random.randint(1, 10, (2, 4))
matrix_multiplication_result = np.dot(C, D)
print("Matrix multiplication result:")
print(matrix_multiplication_result)

# 83. Create a matrix `E` and find its transpose.
E = np.array([[1, 2, 3], [4, 5, 6]])
transpose_E = E.T
print("Matrix E:")
print(E)
print("Transposed matrix E:")
print(transpose_E)

# 84. Generate a square matrix `F` and compute its determinant.
F = np.random.randint(1, 10, (3, 3))
determinant_F = np.linalg.det(F)
print("Matrix F:")
print(F)
print("Determinant of matrix F:", determinant_F)

# 85. Create a square matrix `G` and find its inverse.
G = np.array([[1, 2], [3, 4]])
inverse_G = np.linalg.inv(G)
print("Matrix G:")
print(G)
print("Inverse of matrix G:")
print(inverse_G)

