import h5py

file_path = 'dataset/iclr_final_truncated_fixed_powers.h5'

with h5py.File(file_path, 'r') as f:
    print("all Dataset", list(f.keys()))
    
    dataset_name = list(f.keys())[0]
    data = f[dataset_name]
    print(f"DATASET {dataset_name} Shape: {data.shape}")
