from ulab import numpy as np
def csv_read(file_name):  # function for reading csv file, return file content as numpy array
    f = open(file_name, 'r')
    w = []
    tmp = []
    for each in f:
        w.append(each)
        # print (each)

    # print(w)
    for i in range(len(w)):
        data = w[i].split(",")
        tmp.append(data)
        # print(data)
    file_data = np.array(([[float(y) for y in x] for x in tmp]))
    #file_data = [[float(y) for y in x] for x in tmp]
    return file_data
 
