import numpy as np
import os

def main():
    vals = []
    # lookup = list(range(0,790,10))
    path_list = []
    for filename in os.listdir('./raycast_lookup/'):
        if filename.endswith('.npz'):
            path_list.append(filename)

    path_list.sort()

    for path in path_list:
        np_obj = np.load(os.path.join('./raycast_lookup', path))
        vals.append(np_obj['arr'])

    lookup_table = np.vstack(vals)
    name = os.path.join('./raycast_lookup', 'complete')
    np.savez(name, arr=lookup_table)

    print(lookup_table.shape)

if __name__ == '__main__':
    main()