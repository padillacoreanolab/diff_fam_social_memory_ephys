class LFPRecording:
    def __init__(
        self,
        recording_to_behavior_dict,
        subject_to_channel_dict,
        data_path,
        elec_noise_freq=60,
        sampling_rate=20000,
        min_freq=0.5,
        max_freq=300,
        resample_rate=1000,
        voltage_scaling=0.195,
        spike_gadgets_multiplier=0.675,
        threshold=None,
        halfbandwidth=2,
        timewindow=1,
        timestep=0.5,
    ):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.recording_to_behavior_dict = recording_to_behavior_dict
        self.subject_to_channel_dict = subject_to_channel_dict
        self.voltage_scaling = voltage_scaling
        self.spike_gadgets_multiplier = spike_gadgets_multiplier
        self.elec_noise_freq = elec_noise_freq
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resample_rate = resample_rate
        self.threshold = threshold
        self.halfbandwidth = halfbandwidth
        self.timewindow = timewindow
        self.timestep = timestep
        self.make_recordings()

    def make_recordings():
        return
