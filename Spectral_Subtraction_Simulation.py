import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#creates a generator for random numbers
rng = np.random.default_rng()

#creates an a time interval for smooth interpolation
sampling_interval = 0.000125
sampling_time = np.arange(0, 10 + sampling_interval, sampling_interval)

#A base frequency of 1 is easier to see on the graphs, but this value can be modified
base_frequency = 1

signal_1 = lambda t: np.sin(base_frequency * np.pi *t) + np.cos(base_frequency * 2 * np.pi * t)
signal_2 = lambda t: np.sin(base_frequency * 6 * np.pi *t)

#creates a normal distribution of random numbers centered around 0
random_data = rng.normal(loc=0.0, scale=1, size=sampling_time.size)
start_signal = 1

#makes the first second of the signal silent
signal_delay = np.where(sampling_time > start_signal, signal_1(sampling_time) + signal_2(sampling_time), 0)

audio = signal_delay + random_data

#perform a short time fourier transform on the mixed signal. The sampling frequency is rather high, but it passes the Nyquist criterion
frequency, time, data = signal.stft(audio, 8000, nperseg=1024)
magnitude = np.abs(data)

#finds the index which corresponds to the first second
end_index = np.searchsorted(time, 1.0)

#average the noise to create a noise floor
noise_floor = np.mean(magnitude[:, :end_index], axis=1)

#set the subtraction constant
alpha = 3.0 

#the noise floor needs to be recast in order for subtraction to work
subtraction = magnitude - (noise_floor.reshape(-1, 1) * alpha)
clean_magnitude = np.maximum(subtraction, 0)
phase = np.angle(data)

#recover the estimated signal
clean_data = clean_magnitude * np.exp(1j * phase)
time_clean, clean_audio = signal.istft(clean_data, 8000, nperseg=1024)

plt.figure(figsize=(10, 6))

plt.subplot(4, 1, 1)
plt.title("Original Signal $x(t) = sin(\pi*t) + sin(6\pi*t) + cos(2\pi*t)$")
plt.plot(sampling_time, signal_delay, color='green')

plt.subplot(4, 1, 2)
plt.title("Original Signal (Noise and 2 Sine Waves)")
plt.plot(sampling_time, audio, color='red', alpha=0.7)

plt.subplot(4, 1, 3)
plt.title("Cleaned Signal (Spectral Subtraction Output)")
plt.plot(time_clean, clean_audio, color='blue')

plt.subplot(4, 1, 4)
plt.title("Random noise")
plt.plot(sampling_time, random_data, color='blue')

plt.tight_layout()
plt.show()
