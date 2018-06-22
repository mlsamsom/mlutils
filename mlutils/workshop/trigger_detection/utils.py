import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt


def preprocess_audio(filename, duration=10000, frame_rate=44100):
    """Preprocess audio samples to correct format

    Resaves the audio file in the same locations

    Args:
        filename (str): location of wav file
        duration (int): number of samples to make the segment
        frame_rate (int): frame rate of sampling
    """

    # Trim or pad audio segment
    padding = AudioSegment.silent(duration=duration)
    segment = AudioSegment.from_wav(filename)[:duration]
    segment = padding.overlay(segment)

    # Set frame rate to 44100
    segment = segment.set_frame_rate(frame_rate)

    # Export as wav
    segment.export(filename, format='wav')


def create_training_example(background, activates, negatives, output_name, Ty, show=False):
    """Create training samples

    Creates a training example with a given background, activates, and negatives.

    Args:
        background (ndarray): a 10 second background audio recording
        activates (ndarray): a list of audio segments of the word "activate"
        negatives (ndarray): a list of audio segments of random words that are not "activate"
        output_name (str): name to save training example

    Returns:
        x (ndarray): the spectrogram of the training example
        y (ndarray): the label at each time step of the spectrogram
    """

    # Set the random seed
    np.random.seed(18)

    # Make background quieter
    background = background - 20

    y = np.zeros((1, Ty))

    previous_segments = []

    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end, Ty)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        # Insert the audio clip on the background
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)

    # Export new training example
    file_handle = background.export(output_name, format="wav")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram(output_name, show=show)

    return x, y


def insert_ones(y, segment_end_ms, Ty, steps=50, background_len=10000.0):
    """Update the label vector y

    The labels of the output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.

    Args:
        y (ndarray): numpy array of shape (1, Ty), the labels of the training example
        segment_end_ms (int): the end time of the segment in ms
        steps (int): number of output steps after to segment to put the label
        background_len (float): number of time steps in the sample

    Returns:
        y (ndarray): updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / background_len)

    for i in range(segment_end_y+1, segment_end_y+steps+1):
        if i < Ty:
            y[0, i] = 1.0

    return y


def insert_audio_clip(background, audio_clip, previous_segments):
    """Insert the audio segment over background noise

    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Args:
        background (ndarray): a 10 second background audio recording.
        audio_clip (ndarray): the audio clip to be inserted/overlaid.
        previous_segments (ndarray): times where audio segments have already been placed

    Returns:
        new_background (ndarray): the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)

    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)

    # Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])

    return new_background, segment_time


def is_overlapping(segment_time, previous_segments):
    """Check for time overlap

    Checks if the time of a segment overlaps with the times of existing segments.

    Args:
        segment_time (tuple): a tuple of (segment_start, segment_end) for the new segment
        previous_segments (tuple): a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
        overlap (bool): True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    overlap = False

    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
    return overlap


def get_random_time_segment(segment_ms, duration=10000):
    """Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Args:
        segment_ms (int): the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
        segment_time (tuple): a tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0, high=duration-segment_ms)
    segment_end = segment_start + segment_ms - 1

    return segment_start, segment_end


def load_raw_audio(data_dir):
    """Load raw audio
    """
    activates_path = os.path.join(data_dir, "activates")
    backgrounds_path = os.path.join(data_dir, "backgrounds")
    negatives_path = os.path.join(data_dir, "negatives")

    activates = []  # positive samples
    backgrounds = []
    negatives = []
    for fn in os.listdir(activates_path):
        if fn.endswith("wav"):
            activate = AudioSegment.from_wav(
                os.path.join(activates_path, fn)
            )
            activates.append(activate)

    for fn in os.listdir(backgrounds_path):
        if fn.endswith("wav"):
            background = AudioSegment.from_wav(
                os.path.join(backgrounds_path, fn)
            )
            backgrounds.append(background)

    for fn in os.listdir(negatives_path):
        if fn.endswith("wav"):
            negative = AudioSegment.from_wav(
                os.path.join(negatives_path, fn)
            )
            negatives.append(negative)

    return activates, negatives, backgrounds


def get_wav_info(wav_file):
    """Load a wav file
    """
    rate, data = wavfile.read(wav_file)
    return rate, data


def match_target_amplitude(sound, target_dBFS):
    """Standardize volume of audio clip
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def graph_spectrogram(wav_file, len_win_seg=200, fs=8000, overlap=100, show=False):
    """Calculate and plot a spectrogram

    Args:
        wav_file (str): path to wav file
        len_win_seg (int): Length of each window segment
        fs (int): Sampling frequency
        overlap (int): Overlap between windows
        show (bool): Whether to show the plot

    Returns:
        pxx (ndarray): spectrogram
    """
    rate, data = get_wav_info(wav_file)
    nchannels = data.ndim

    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, len_win_seg, fs, noverlap=overlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], len_win_seg, fs, noverlap=overlap)
    else:
        raise RuntimeError("unexpected number of channels")
    if show:
        plt.show()
    return pxx
