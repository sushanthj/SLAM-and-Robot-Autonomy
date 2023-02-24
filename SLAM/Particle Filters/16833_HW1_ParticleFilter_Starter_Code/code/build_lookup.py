import numpy as np
import os

def main():
    vals = []
    # lookup = list(range(0,790,10))
    file_list = []
    for filename in os.listdir('./raycast_lookup/'):
        if filename.endswith('.npz'):
            file_list.append(int(filename.split('.npz')[0]))

    file_list.sort()
    print(file_list)

    for file in file_list:
        full_filename = str(file) + '.npz'
        np_obj = np.load(os.path.join('./raycast_lookup', full_filename))
        vals.append(np_obj['arr'])

    lookup_table = np.vstack(vals)
    name = os.path.join('./raycast_lookup', 'complete')
    np.savez(name, arr=lookup_table)

    print(lookup_table.shape)

if __name__ == '__main__':
    main()