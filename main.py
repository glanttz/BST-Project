import numpy as np
from array import array
from scipy.io.wavfile import read as wavread
from scipy.stats import entropy as en
import pandas as pd
from math import log, e

import matplotlib.pyplot as plt

samplerate = 44100
freq = 440
seconds = 2
time = np.linspace(0., seconds, seconds * samplerate, endpoint=False)
signal = np.sin(2 * np.pi * freq * time)

rate, data = wavread('samples.wav')
print('rate:', rate, 'Hz')
print('data is a:', type(data))
print('data shape is:', data.shape)



plt.subplots(dpi=100)
plt.plot(time[:200] * 1000, data[:200])
plt.xlabel('milliseconds')
plt.ylabel('signed 16-bit integers')
plt.show()

results = array('L', [])

result = 0
flag = 0
for i in np.array(data[:, 1]):
    temp = i & 1
    result = (result | temp) << 1
    # result = result << 1
    flag += 1
    # print("flag:", flag)
    # print(result)
    if flag == 7:
        results.append(result)
        # print("result:", result)
        # print(i, bin(result))
        flag = 0
        result = 0

print(results.tolist())

# for i in range(0, len(data)):
#     print(data[i])

# plt.hist(data, bins="auto")
# plt.show()


fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(results, bins=40)
plt.show()
#data = results.value_counts()
#en(data)

results = pd.Series(results)
data = results.value_counts()
print('entropy1:', en(data))

def entropy3(results, base=None):
  vc = pd.Series(results).value_counts(normalize=True, sort=False)
  base = e if base is None else base
  return -(vc * np.log(vc)/np.log(base)).sum()
print('entropy3:' , entropy3(results))

def entropy4(results, base=None):
  value,counts = np.unique(results, return_counts=True)
  norm_counts = counts / counts.sum()
  base = e if base is None else base
  return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

print('entropy4:', entropy4(results))