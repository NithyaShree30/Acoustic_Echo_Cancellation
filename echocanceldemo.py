import numpy as np
import matplotlib.pyplot as plt
import adaptfilt as adf
import winsound
from scipy.io import wavfile

waveout = 'C:/Users/ashwi/Desktop/College/Sem-8/Final-Project/Audio-Samples/output.wav'#/echo-cancel-master/output.wav' # Defining the output wave file
step = 0.04 # Step size
M = 45 # Number of filter taps in adaptive filter

# Read the audio files of sender and listener
sfs, u = wavfile.read('C:/Users/ashwi/Desktop/College/Sem-8/Final-Project/Audio-Samples/sub_lpb.wav')#/echo-cancel-master/sender.wav')
lfs, v = wavfile.read('C:/Users/ashwi/Desktop/College/Sem-8/Final-Project/Audio-Samples/sub_mic.wav')#/echo-cancel-master/listener.wav')

u = np.frombuffer(u, np.int16)
u = np.float64(u)

v = np.frombuffer(v, np.int16)
v = np.float64(v)

# Generate the feedback signal d(n) by:
# a) convolving the sender's voice with randomly chosen coefficients assumed to emulate the listener's room characteristic, and
# b) mixing the result with listener's voice, so that the sender hears a mix of noise and echo in the reply.

'''coeffs = np.concatenate(([0.8], np.zeros(8), [-0.7], np.zeros(9),
                         [0.5], np.zeros(11), [-0.3], np.zeros(3),
                         [0.1], np.zeros(20), [-0.05]))'''

coeffs = [0.3,0.2,0.1,0.5,0.3]

d = np.convolve(u, coeffs)
d = d/20.0
v = v/20.0
d = d[:len(v)] # Trims sender's audio to the same length as that of the listener's in order to mix them
d = d + v - (d*v)/256.0   # Mix with listener's voice.
d = np.round(d,0)

# Hear how the mixed signal sounds before proceeding with the filtering.
dsound = d.clip(min=-1, max=1)
dsound = d.astype('int16')
wavfile.write(waveout, lfs, dsound)
winsound.PlaySound(waveout, winsound.SND_ALIAS)

# Apply adaptive filter
#y, e, w = adf.nlms(u[:len(d)], d, M, step, returnCoeffs=True)
e, H = adf.kalman(u, d)

# The algorithm stores the processed result in the variable 'e', which is the mix of the error signal and the listener's voice.
# Hear how e sounds now.  Ideally we on behalf of the sender, should hear only the listener's voice.  Practically, some echo would still be present.

# Hear how the mixed signal sounds after filtering.
e = e.astype('int16')
wavfile.write(waveout, lfs, e)
winsound.PlaySound(waveout, winsound.SND_ALIAS)

fig, (p1, p2, p3, p4) = plt.subplots(4, 1)
p1.plot(u, color='blue')
p2.plot(v, color='red')
p3.plot(d, color='green')
p4.plot(e, color='purple')
fig.suptitle('Sender, Listener, Feedback and Enhanced Signals', fontsize=16)
fig.supxlabel('Samples', fontsize=14)
fig.supylabel('Amplitude', fontsize=14)
fig.legend(["Sender", "Listener", "Feedback", "Enhanced"], loc='upper left')
fig.tight_layout(pad=0.3)

# Calculate and plot the mean square weight error
#mswe = adf.mswe(w, coeffs)
mswe = adf.mswe(H, coeffs)
plt.figure(2)
plt.title('Mean Squared Weight Error', fontsize=16)
plt.plot(mswe)
plt.grid()
plt.xlabel('Samples', fontsize=14, labelpad=10)
plt.ylabel('MSWE', fontsize=14, labelpad=20)
plt.show()