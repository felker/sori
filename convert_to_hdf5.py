import numpy as np
import h5py

shot_num=186121
shot_data_time = np.genfromtxt('{shot_num}.csv', delimiter=',')
shot_data=shot_data_time[:,0:8]
times=shot_data_time[:,8]

print(len(times))
n_features=8
n_times=21600

hf = h5py.File(f'{shot_num}.h5', 'w')

#g1 = hf.create_group('/')
g1=hf
g1.create_dataset('X',data=shot_data, dtype='f4')
g1.create_dataset('n_features',data=n_features, dtype='i4')
g1.create_dataset('n_times',data=n_times, dtype='i4')
g1.create_dataset('time',data=times, dtype='f4')

hf.close()
