from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from FR3.src.observations.band import Band, Bands
from FR3.src.methods.music import FR3_MUSIC
from FR3.src.observations.ny_dataset import NYDataset
from FR3.utils.constants import *
from FR3.utils.global_functions import set_unified_seed, optimize_to_estimate_position

def main():
    set_unified_seed()

    # Initialize the channel and band
    fcs = [6, 12, 18, 24]
    bws = [6, 12, 24, 48]
    ns = [4, 8, 16, 24]
    ks = [50, 100, 75, 100]

    bands = Bands(fcs, ns, ks, bws)
    data_dir = r"C:\git_repos\DCD-MUSIC\FR3\data\FR3"
    dataset = NYDataset(data_dir, bands)
    # dataset.load(r"C:\git_repos\DCD-MUSIC\FR3\src\observations\test.npy")
    dataset.create_data()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for data, gt in dataloader:
        break
    # estimate the physical parameters
    # create instance of the algorithm
    music = FR3_MUSIC(bands)
    # estimate the physical parameters
    angle, time_delay, power = music(data)
    print(f"Estimated - Angle: {angle}, Time delay: {time_delay}, Power: {power}")
    print(f"Ground truth - Angle: {gt['angle'].numpy()}, Time delay: {gt['time_delay'].numpy()}")
    locs = []
    for i in range(len(angle)):
        locs.append(optimize_to_estimate_position(gt['bs_loc'][i], time_delay[i], angle[i], medium_speed=gt['medium_speed'][i].item()))
    print(f"Estimated location: {locs}")
    print(f"Ground truth location: {gt['ue_pos']}")
    print(f"RMSE: {mean_squared_error(gt['ue_pos'], locs, squared=False)}")



if __name__ == "__main__":
    main()
    # TODO: add the proposed solution of Tomer
    # TODO: think of the way to integrate the data to be suitable for the SSN, mainly, how to create the dataset?