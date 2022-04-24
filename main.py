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

# length = data.shape[0] / rate
# time = np.linspace(0., length, data.shape[0])
# plt.plot(time, data[:, 0], label="Left channel")
# plt.plot(time, data[:, 1], label="Right channel")
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()

plt.subplots(dpi=100)
plt.plot(time[:200] * 1000, data[:200])
plt.xlabel('milliseconds')
plt.ylabel('signed 16-bit integers')
plt.show()

results = array('L', [])

fig, axs = plt.subplots(2, 2, figsize=(20,15))

def makeInputData(input, mask):
    inputData = input
    result = 0
    flag = 0
    for i in np.array(data[:, 1]):
        temp = i & mask
        result = (result | temp) << 1
        # result = result << 1
        flag += 1
        # print("flag:", flag)
        # print(result)
        if flag == 7:
            inputData.append(result)
            # print("result:", result)
            # print(i, bin(result))
            flag = 0
            result = 0
    
    return inputData
    
def makeHistogramGraph( data,title, row, col):
    axs[row,col].hist(data, bins = 40)
    axs[row,col].set_title(title)

# print(results.tolist())

# for i in range(0, len(data)):
#     print(data[i])

# plt.hist(data, bins="auto")
# plt.show()


# axs[0, 0].hist(results, bins=40)
# axs[0, 0].set_title('hist')


histOneBit = makeInputData(results, 1)
# hisTwoBits = makeInputData(results, 2)
# hisThreeBits = makeInputData(results, 3)
# histFourBits = makeInputData(results, 4)

makeHistogramGraph(histOneBit, "histogram dla próbek 8 bitowych dla jednego bitu wyekstraktowanego", 0,0)
# makeHistogramGraph(hisTwoBits, "histogram dla próbek 8 bitowych dla dwóch bitów wyekstraktowanego", 0,1)
# makeHistogramGraph(hisThreeBits, "histogram dla próbek 8 bitowych dla trzech bitów wyekstraktowanego", 1,0)
# makeHistogramGraph(histFourBits, "histogram dla próbek 8 bitowych dla czterech bitów wyekstraktowanego", 1,1)


# fig, ax = plt.subplots(figsize=(10, 7))
# ax.hist(results, bins=40)
# plt.show()

#data = results.value_counts()
#en(data)

results = pd.Series(results)
histOneBit = pd.Series(histOneBit)
# hisTwoBits = pd.Series(hisTwoBits)
# hisThreeBits = pd.Series(hisThreeBits)
# histFourBits = pd.Series(histFourBits)
data = results.value_counts()
print('entropy1:', en(data))

# def entropy3(results, base=None):
#   vc = pd.Series(results).value_counts(normalize=True, sort=False)
#   base = e if base is None else base
#   return -(vc * np.log(vc)/np.log(base)).sum()
# print('entropy3:' , entropy3(results))

def entropy4(results, base=None):
  value,counts = np.unique(results, return_counts=True)
  norm_counts = counts / counts.sum()
  base = e if base is None else base
  return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

print('entropia dla próbek z najmlodszego bitu:', entropy4(histOneBit))
# print('entropia dla próbek z  dwóch najmlodszych bitu:', entropy4(hisTwoBits))
# print('entropia dla próbek z  trzech najmlodszych bitu:', entropy4(hisThreeBits))
# print('entropia dla próbek z  czterech najmlodszych bitu:', entropy4(histFourBits))