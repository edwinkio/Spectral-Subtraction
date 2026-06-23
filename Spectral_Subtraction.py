import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

#laod both the noise file and contaminated audio
noise_file, noise_sr = librosa.load("noise.wav")
audio_file, audio_sr = librosa.load("audio.wav")

#ensure both sample rates match. Otherwise, the frequencies wouldn't align in time
assert noise_sr == audio_sr, "Please make sure that the sample rates match!"

noise_frequency = librosa.stft(noise_file, n_fft=512)
audio_frequency = librosa.stft(audio_file, n_fft=512)

average_noise = np.mean(np.abs(noise_frequency), axis=1)

#setting alpha > 1 represents oversubtraction
alpha = 1.0

audio_phase = np.angle(audio_frequency)

#drop the magnitude to 0 in case of negative frequencies
clean_magnitude = np.maximum(np.abs(audio_frequency) - (alpha * average_noise.reshape(-1,1)), 0)

#recover the estimated noise free audio
clean_estimate = 3 * clean_magnitude * np.exp(1j * audio_phase)
clean_audio = librosa.istft(clean_estimate)

sf.write("clean_audio.wav", clean_audio, audio_sr)