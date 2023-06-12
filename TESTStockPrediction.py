import numpy as np

array1 = np.array([[1, 2, 3], [4, 5, 6]])  # 2-dimensional array
array2 = np.array([7, 8])  # 1-dimensional array

# Reshape array2 to have a compatible shape for concatenation
array2_2d = array2[:, np.newaxis]  # Reshape to a 2-dimensional array

# Concatenate array1 and array2_2d along axis 1 (column-wise concatenation)
result = np.concatenate((array1, array2_2d), axis=1)

print(result)
