import time

from FR3.src.observations.observations import Observations
from FR3.src.observations.channel import Channel, Channels
from FR3.src.observations.band import Band, Bands
from FR3.src.methods.music import FR3_MUSIC

from FR3.utils.global_functions import set_unified_seed, optimize_to_estimate_position
from sklearn.metrics import mean_squared_error

def main():
    set_unified_seed()
    ue_pos = [90, 75]

    # Initialize the channel and band
    fcs = [6, 12, 18, 24]
    bws = [6, 12, 24, 48]
    ns = [4, 8, 16, 24]
    ks = [50, 100, 75, 100]
    bands = Bands(fcs, ns, ks, bws)
    channels = Channels(synthetic=False)
    channels.init_channels([1 for i in range(bands.get_band_count())], ue_pos, bands)
    # Initialize the observations
    observations = Observations([])
    for band, channel in zip(bands, channels):
        obser = observations.init_observations(channel, band)
        observations.add_observation(obser)

    # estimate the physical parameters
    # create instance of the algorithm
    music = FR3_MUSIC(bands)
    # estimate the physical parameters
    angle, time_delay, power = music(observations.get_observations())
    print(f"Estimated - Angle: {angle}, Time delay: {time_delay}, Power: {power}")
    print(f"Ground truth - Angle: {channel.get_doas()[0]}, Time delay: {channel.get_toas()[0]}, Power: {channel.get_powers()[0]}")
    loc = optimize_to_estimate_position(channel.get_bs_loc(), time_delay, angle, medium_speed=channel.get_medium_speed())
    print(f"Estimated location: {loc}")
    print(f"Ground truth location: {ue_pos}")
    print(f"RMSE: {mean_squared_error(ue_pos, loc, squared=False)}")



if __name__ == "__main__":
    main()
    # TODO: add the proposed solution of Tomer
    # TODO: think of the way to integrate the data to be suitable for the SSN, mainly, how to create the dataset?