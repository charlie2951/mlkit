"""Helper code to implement numpy functions those are 
not present in existing numpy distro in ulab """
from ulab import numpy as np
import random
import math
import sys
#to be implemented np.divide()

#Implementation of numpy.argsort() function
def np_argsort(arr):
    """
    Returns the indices that would sort an array.

    Parameters:
    arr (array_like): Input array.

    Returns:
    ndarray: Array of indices that sort the input array.
    """
    # Create a list of tuples where each tuple contains the element and its index
    indexed_arr = [(elem, idx) for idx, elem in enumerate(arr)]
    
    # Sort the list of tuples based on the first element of each tuple (the array element)
    sorted_arr = sorted(indexed_arr, key=lambda x: x[0])
    
    # Extract and return the indices from the sorted list
    return [idx for _, idx in sorted_arr]

# Example usage:
if __name__ == "__main__":
    arr = [3, 1, 2, 4]
    sorted_indices = argsort(arr)
    print("Sorted indices:", sorted_indices)


#Function to implement numpy.random.shuffle()
def np_random_shuffle(a):
    """
    Shuffles the elements of an array in-place.

    Args:
        a: The array to shuffle.

    Returns:
        None. The elements of the input array are shuffled in-place.
    """

    for i in range(len(a) - 1, 0, -1):
        # Generate a random index in the range [0, i]
        j = random.randrange(i + 1)

        # Swap the elements at indices i and j
        a[i], a[j] = a[j], a[i]
        
#Function equivalent to numpy.expand_dims()
def np_expand_dims(a, axis=-1):
  """
  Expands the shape of an array by inserting new singleton dimensions at the given axis.

  Args:
    a: The input array.
    axis: The axis along which to insert the new dimensions. Default is -1, which inserts
          at the end.

  Returns:
    A new array with the expanded dimensions.
  """

  a = np.asarray(a)  # Ensure input is a NumPy array
  new_shape = list(a.shape)
  new_shape.insert(axis, 1)  # Insert a new dimension of size 1 at the specified axis
  return a.reshape(tuple(new_shape))  # Reshape the array with the new dimensions

#Implementing Numpy round() funcion from scratch in micropython
#Implementing Numpy round() funcion from scratch in micropython
def np_round(arr, decimal_digits=0):#input numpy array 1d
    arr = arr.tolist() 	
    rounded_list = [round(x, decimal_digits) for x in arr]
    return np.array(rounded_list)

#Function to implement np.atleast_1d()
def np_atleast_1d(*arrays):
    """
    Converts inputs to arrays with at least one dimension.

    Args:
        *arrays: One or more input arrays or scalars.

    Returns:
        A tuple of arrays, each with at least one dimension.
    """

    result = []
    for array in arrays:
        array = np.array(array)  # Ensure array-like inputs are converted to NumPy arrays
        if array.ndim == 0:
            result.append(array[np.newaxis])  # Add a new axis for scalars
        else:
            result.append(array)  # Preserve higher-dimensional arrays

    return tuple(result)  # Return a tuple to match NumPy's behavior

  
#function implementing numpy.power()
def np_power(base, exponent):
    """
    Raises the elements of `base` to the powers in `exponent` element-wise.

    Args:
        base (array_like): The base array.
        exponent (array_like): The exponent array.

    Returns:
        ndarray: The result of raising the elements of `base` to the powers in `exponent`.

    Raises:
        ValueError: If `base` and `exponent` have different shapes.
    """

    if base.shape != exponent.shape:
        raise ValueError("base and exponent must have the same shape")

    result = np.empty_like(base)
    for i in np.ndindex(base.shape):
        result[i] = base[i] ** exponent[i]

    return result
    
#code to implement numpy.unique() with return count
def np_unique(arr, return_counts=True):
  """
  Implements a custom version of numpy.unique() with return_counts=True functionality.

  Args:
    arr: A NumPy array of values.

  Returns:
    A tuple containing:
      unique: A NumPy array of unique values in the input array.
      counts: A NumPy array of counts for each unique value.
  """

  # Initialize empty lists to store unique values and counts
  unique_values = []
  counts = []

  # Iterate through the array
  for value in arr:
    # Check if the value is already in the unique_values list
    if value not in unique_values:
      unique_values.append(value)
      counts.append(1)
    else:
      # Increment the count for the existing value
      index = unique_values.index(value)
      counts[index] += 1

  # Convert lists to NumPy arrays
  unique = np.array(unique_values)
  counts = np.array(counts)
  if(return_counts==True):
    return unique, counts
  else:
    return unique
    
#Implementation of numpy.bincount()
def np_bincount(arr):
  """
  Counts the occurrences of each value in an array.

  Args:
      arr (array_like): The input array.

  Returns:
      ndarray: The array of counts, where the index corresponds to the value and the value is the count.
  """

  # Handle negative values by adding a constant offset
  min_value = np.min(arr)
  if min_value < 0:
    arr += abs(min_value) + 1

  # Create an empty array to store counts
  counts = np.zeros(int(np.max(arr) + 1))

  # Iterate through the array and increment the corresponding count
  for value in arr:
    counts[int(value)] += 1

  return counts 
