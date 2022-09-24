import numpy as np
import numpy_methods as npm
import ehlers_filters as ehlers
import matplotlib.pyplot as plt
from nptyping import NDArray, Shape, Float
from dataclasses import dataclass
from ehlers_indicators import AGC

OneDimensionalFloatArray = NDArray[Shape['Any'], Float]

@dataclass
class Filtering:
    price: OneDimensionalFloatArray
    fs: int = 1
    pc_lp: int = 10
    pc_hp: int = 48
    
    def __set_filters__(self):
        self.supersmoother = lambda y: ehlers.SuperSmoother(y, self.fs, self.pc_lp)
        self.hp2 = lambda y: ehlers.EhlersSecondOrderHP(y, self.fs, self.pc_hp)
        self.roof = lambda y: ehlers.EhlersRoofingFilter(y, self.fs, self.pc_lp, self.pc_hp)

@dataclass
class Ploting:
    def plotsDC(self):
        fig, axes = plt.subplots(2, 1,figsize=(16,9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        axes[0].set_title("Price")
        axes[0].plot(self.price)
        axes[0].grid()
        
        axes[1].set_title("Dominant Cycle Period")
        axes[1].plot(self.dcs, color="green")
        axes[1].set_yticks([10,20,30,40,50])
        axes[1].grid()

@dataclass
class ZeroCrossingsDC(Filtering, Ploting):   
    def __post_init__(self):
        self.__set_filters__()

        filt = self.roof(self.price).output()
        peaks = AGC(filt)
        signal = filt/peaks
        
        signal_1 = npm.nanshift(signal,1)
        zerocross = np.sign(signal*signal_1) == -1
        
        self.dcs = np.zeros_like(self.price)
        dc = 0
        counter = 0
        for i, has_crossed in enumerate(zerocross):
            counter += 1
            if dc < 6:
                dc = 6
            if has_crossed:
                dc = 2*counter
                if 2*counter > 1.25*dc:
                    dc = 1.25*dc
                elif 2*counter < 0.8*dc:
                    dc = 0.8*dc
                counter = 0
            self.dcs[i] = dc

@dataclass
class AutocorrelationPeriodogram(Filtering, Ploting):
    lags_range: OneDimensionalFloatArray = np.arange(6, 61)
    bars: int = 48

    def __post_init__(self):
        self.__set_filters__()

        self.periods = np.arange(10, 49, dtype=int)
        self._getDC()
    
    def _getR(self) -> OneDimensionalFloatArray:
        x = -self.bars - self.day
        y = self.price.size - self.day
        
        autocorr = np.array([np.corrcoef(self.price[x:y], self.price[x-lag:y-lag])[0,1] for lag in self.lags_range])[:,None]
        
        wave_arg = 2*np.pi*self.lags_range[:,None]/self.periods[None,:]
        cosines = np.cos(wave_arg)
        sines = np.sin(wave_arg)
        
        self.real_part = (autocorr*cosines).sum(axis=0)
        self.imag_part = (autocorr*sines).sum(axis=0)
        
        sqsum = self.real_part**2 + self.imag_part**2
        R = sqsum**2
        R = self.supersmoother(R).output()
        return R

    def _getDC(self):
        maxpwr = 0
        self.dcs = np.zeros_like(self.price)
        self.dcs.fill(np.nan)
        valid_range = range( self.price.size - self.bars - self.lags_range.max() )
        for i, self.day in enumerate(valid_range):
            maxpwr *= 0.995
            R = self._getR()
            maxpwr = max(maxpwr, R.max())
            pwr = R / maxpwr
            dc = (self.periods*pwr).sum() / pwr.sum()
            self.dcs[-i-1] = dc

@dataclass
class MESA(Filtering, Ploting):
    m: int = 15
    window: int = 30
    fs: int = 1
    p1: int = 10
    p2: int = 48
    showplots: bool = False
    
    def __post_init__(self):
        self.__set_filters__()
        self.filt = self.roof(self.price).output()
        
        self._getDC()
    
    def _memcof(self):
        n = self.window
        wk1 = np.zeros(n)
        wk2 = np.zeros(n)
        wkm = np.zeros(self.m)
        d = np.zeros(self.m)
        
        p = np.sum(self.samples**2)
        xms = p / n
        wk1[:n - 1] = self.samples[:n - 1]
        wk2[:n - 1] = self.samples[1 : n]
        
        for k in range(self.m):
            num = np.sum(wk1[:n - k - 1] * wk2[:n - k - 1])
            denom = np.sum(wk1[:n - k - 1]**2 + wk2[:n - k - 1]**2)
            
            d[k] = 2.0*num / denom
            xms *= 1.0 - d[k]**2
            
            d[:k] = wkm[:k] - d[k]*wkm[:k][::-1]
                        
            if k != self.m - 1:
                wkm[:k+1] = d[:k+1]        
                for j in range(n-k-2):
                    wk1[j] -= (wkm[k]*wk2[j])
                    wk2[j] = wk2[j+1]-wkm[k]*wk1[j+1]
            
        self.xms = xms
        self.d = d[:,None]
    
    def _spectrum(self):
        self.p_range = np.linspace(self.p1, self.p2, 512)
        w_range = 2*np.pi/self.p_range

        z = np.exp(1j*w_range)
        zetas = np.array([z**(-i-1) for i in range(self.m)])
        denom = np.absolute( 1 - np.sum(self.d * zetas, axis=0) )**2
        self.power = self.xms / denom
        self.dbpowergain = 10*np.log10(np.abs(self.power))
        
    def _getDC(self):
        self.dcs = np.zeros_like(self.filt)
        self.dcs.fill(np.nan)
        
        for i in range(self.filt.size - self.window):
            self.samples = self.filt[-self.window - i : self.filt.size - i]
            self._memcof()
            self._spectrum()
            
            dc = (self.p_range * self.power).sum() / self.power.sum()
            self.dcs[-i-1] = dc
        
        if self.showplots:
            plt.plot(self.p_range, self.dbpowergain) # bars/cycle (Period), Log plot
            
            inf_lim = self.p1 - 4
            sup_lim = self.p2 + 4
            plt.xlim(inf_lim,sup_lim)
            
            beginning = round(inf_lim - 5, -1)
            end = round(sup_lim + 5, -1)
            
            arr_xticks = np.arange(start=beginning, stop=1 + end, step=10)
            arr_xticks_str = np.array(arr_xticks, dtype=str)
            
            plt.xticks(arr_xticks, arr_xticks_str)
            plt.xlabel('Period (bars/cycle)')
            plt.ylabel('Gain (dB)')
            plt.title('MESA')
            plt.grid()
            plt.show()