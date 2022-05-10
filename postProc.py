from array import array
from scipy.io.wavfile import read as wavread
from scipy.stats import entropy as en
import pandas as pd
from math import log, e
from posixpath import split
import sys 
import hashlib
from ast import For
import numpy as np
import matplotlib.pyplot as plt
import math


samplerate = 44100
freq = 440
seconds = 2
time = np.linspace(0., seconds, seconds * samplerate, endpoint=False)
signal = np.sin(2 * np.pi * freq * time)

rate, data = wavread('samples.wav')
print('rate:', rate, 'Hz')
print('data is a:', type(data))
print('data shape is:', data.shape)

# plt.subplots(dpi=100)
# plt.plot(time[:200] * 1000, data[:200])
# plt.xlabel('milliseconds')
# plt.ylabel('signed 16-bit integers')
# plt.show()

results = array('L', [])

fig, axs = plt.subplots(2, 2, figsize=(20,15))

def makeInputData(data, val = 1):
    inputData = []
    result = 0
    mask = 0b00000000
    shifter = 0
    divider = 0
    divRes = 0
    if val == 1:
        mask = 0b00000001
        shifter = 1
        divider = 8
        divRes = 7
    elif val == 2:
        mask = 0b00000011
        shifter = 2
        divider = 4
        divRes = 3
    elif val == 4:
        mask = 0b00001111
        shifter = 4
        divider = 2
        divRes = 1
    else:
        return
    for idx, item in enumerate(data[:, 0]):
        result = (result << shifter | (item & mask))
        if idx % divider == divRes:
            inputData.append(result)
            result = 0
    return inputData

# def makeHistogramGraph( data,title, row, col):
#     axs[row,col].hist(data, bins = 256)
#     axs[row,col].set_title(title)


BUF_SIZE = 160  

output = []



histOneBit = makeInputData(data, 1)

for item in histOneBit:
    output.append(hashlib.sha1(item))



output2 = [ s.hexdigest() for s in output ] #Convert hash back to equivalent string in hex
split_strings = []
for a_string in output2:
    split_strings.append( [a_string[index : index + 2] for index in range(0, len(a_string), 2)] )
output3 = []
for i in split_strings:
    for j in i:
        output3.append(j)

output4 = [ int(str(s), 16) for s in output3 ]       #Convert hex string to int



# # hisTwoBits = makeInputData(data, 2)

# # histFourBits = makeInputData(data, 4)

# # makeHistogramGraph(histOneBit, "histogram dla próbek 8 bitowych dla jednego bitu wyekstraktowanego", 0,0)
# # makeHistogramGraph(hisTwoBits, "histogram dla próbek 8 bitowych dla dwóch bitów wyekstraktowanego", 0,1)
# # makeHistogramGraph(hisThreeBits, "histogram dla próbek 8 bitowych dla trzech bitów wyekstraktowanego", 1,0)
# # makeHistogramGraph(histFourBits, "histogram dla próbek 8 bitowych dla czterech bitów wyekstraktowanego", 1,1)


# results = pd.Series(results)
# histOneBit = pd.Series(histOneBit)
# # hisTwoBits = pd.Series(hisTwoBits)
# # hisThreeBits = pd.Series(hisThreeBits)
# # histFourBits = pd.Series(histFourBits)
# data = results.value_counts()
# print('entropy1:', en(data))




def entropy1(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  norm_counts = counts / counts.sum()
  base = 2 if base is None else base
  return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()


print(entropy1(output4, 2))
plt.hist(output4, bins=[0, 256], density=True)
plt.show()

# print('entropia dla próbek z najmlodszego bitu:', entropy4(histOneBit))
# print('entropia dla próbek z najmlodszego bitu po processingu:', entropy1(output4))

# # print('entropia dla próbek z  dwóch najmlodszych bitu:', entropy4(hisTwoBits))
# # print('entropia dla próbek z  trzech najmlodszych bitu:', entropy4(hisThreeBits))
# # print('entropia dla próbek z  czterech najmlodszych bitu:', entropy4(histFourBits))