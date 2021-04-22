import numpy as np
import scipy.signal as signal
from scipy import interpolate

class EEMD:
    def __init__(self, amp=400, times=10, n_modes=11, sd=0.2):
        self.amp = amp
        self.times = times
        self.n_modes = n_modes
        self.sd = sd

    def addNoise(self, x):
        length = len(x)
        return x + (np.random.randn(length) * self.amp)
    
    def isMonotonic(self, x):
        max_peaks = signal.argrelextrema(x, np.greater)[0]
        min_peaks = signal.argrelextrema(x, np.less)[0]
        all_num = len(max_peaks) + len(min_peaks)
        if all_num > 0:
            return False
        else:
            return True
    
    def findPeaks(self, x):
        return signal.argrelextrema(x, np.greater)[0]
    
    def isIMF(self, x):
        N = np.size(x)
        pass_zero = np.sum(x[0:N - 2] * x[1:N - 1] < 0)
        peaks_num = np.size(self.findPeaks(x)) + np.size(self.findPeaks(-x))
        if abs(pass_zero - peaks_num) > 1:
            return False
        else:
            return True
    
    def mirror(self, x):
        reverse_x = x[::-1][1:-1]
        return np.concatenate((reverse_x, x, reverse_x))
    
    def getSpline(self, x):
        N = np.size(x)
        peaks = self.findPeaks(x)
        if len(peaks) <= 3:
            if len(peaks) < 2:
                peaks = np.concatenate(([0], peaks))
                peaks = np.concatenate((peaks, [N - 1]))
            t = interpolate.splrep(peaks, y=x[peaks], w=None, xb=None, xe=None, k=len(peaks) - 1)
            return interpolate.splev(np.arange(N), t)
        t = interpolate.splrep(peaks, y=x[peaks])
        return interpolate.splev(np.arange(N), t)
    
    def emd(self, x):
        imf = []
        length = len(x) - 2
        while not self.isMonotonic(x):
            x1 = x
            sd = np.inf
            while sd > self.sd or (not self.isIMF(x1)):
                x1 = self.mirror(x1)
                s1 = self.getSpline(x1)
                s2 = -self.getSpline(-1 * x1)
                x2 = x1 - (s1 + s2) / 2
                x1 = x1[length:-length]
                x2 = x2[length:-length]
                sd = np.sum((x1 - x2) ** 2) / np.sum(x1 ** 2)
                x1 = x2
            imf.append(x1)
            x = x - x1
        imf.append(x)
        return imf
    
    def eemd(self, x):
        x = x.reshape(-1)
        imf, n = 0, 0
        for _ in range(self.times):
            modes = self.emd(self.addNoise(x))
            if len(modes) == self.n_modes:
                imf += np.array(modes)
                n += 1
        print(f'Number of EMDs: {n}')
        length = len(imf)
        for i in range(length):
            imf[i] = imf[i] / n
        return imf
    

if __name__ == '__main__':
    import concurrent.futures
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv('bdi.csv').fillna(method='ffill')['bdi'].values
    index = [data[:i] for i in range(6088, 7589)]

    Decomposition = EEMD(amp=200, times=1000, n_modes=11, sd=0.2)
    with concurrent.futures.ProcessPoolExecutor() as Executor:
        results = Executor.map(Decomposition.eemd, index)
        for index, result in enumerate(results):
            np.save(f'EEMD/EEMD{index:04d}', result)

