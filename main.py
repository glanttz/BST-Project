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

plt.subplots(dpi=100)
plt.plot(time[:200] * 1000, data[:200])
plt.xlabel('milliseconds')
plt.ylabel('signed 16-bit integers')
plt.show()

results = array('L', [])

fig, axs = plt.subplots(2, 2, figsize=(20,15))

# define algorithm to extract the youngest bits

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


# print histogram
def makeHistogramGraph( data,title, row, col):
    axs[row,col].hist(data, bins = 256)
    axs[row,col].set_title(title)



hashOutput = []

histOneBit = makeInputData(data, 1)

# hash function SHA1
for item in histOneBit:
    hashOutput.append(hashlib.sha1(item))


# converting from string to integer
hashOutputConvert = [ s.hexdigest() for s in hashOutput ]

hashOutputHexToInt = []

split_strings = []

for a_string in hashOutputConvert:
    split_strings.append( [a_string[index : index + 2] for index in range(0, len(a_string), 2)] )


for i in split_strings:
    for j in i:
        hashOutputHexToInt.append(j)

results = [ int(str(s), 16) for s in hashOutputHexToInt ]       


hisTwoBits = makeInputData(data, 2)

histFourBits = makeInputData(data, 4)

makeHistogramGraph(histOneBit, "histogram dla próbek 8 bitowych dla jednego bitu wyekstraktowanego", 0,0)
makeHistogramGraph(hisTwoBits, "histogram dla próbek 8 bitowych dla dwóch bitów wyekstraktowanego", 0,1)
makeHistogramGraph(results, "histogram dla próbek 8 bitowych dla 1 bitu wyekstraktowanego po post processingu za pomocą szyfrowania SHA1", 1,0)
makeHistogramGraph(histFourBits, "histogram dla próbek 8 bitowych dla czterech bitów wyekstraktowanego", 1,1)


# results = pd.Series(results)
# histOneBit = pd.Series(histOneBit)
# # hisTwoBits = pd.Series(hisTwoBits)
# # hisThreeBits = pd.Series(hisThreeBits)
# # histFourBits = pd.Series(histFourBits)
# data = results.value_counts()
# print('entropyOperation:', en(data))


def entropyOperation(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  norm_counts = counts / counts.sum()
  base = 2 if base is None else base
  return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()


plt.hist(results, bins=[0, 256], density=True)
plt.show()

print('entropia dla próbek z najmlodszego bitu:', entropyOperation(histOneBit, 2))
print('entropia dla próbek z najmlodszego bitu po processingu:', entropyOperation(results, 2))
print('entropia dla próbek z  dwóch najmlodszych bitu:', entropyOperation(hisTwoBits,2))
print('entropia dla próbek z  czterech najmlodszych bitu:', entropyOperation(histFourBits))