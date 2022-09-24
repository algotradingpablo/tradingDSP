import numpy as np
import numpy_methods as npm
import ehlers_filters as ehlers
from nptyping import NDArray, Shape, Float
from dataclasses import dataclass

OneDimensionalFloatArray = NDArray[Shape['Any'], Float]

def Position(long:OneDimensionalFloatArray, short:OneDimensionalFloatArray) -> OneDimensionalFloatArray:
    position = np.zeros_like(long)
    position = np.where(long, 1, np.nan)
    position = np.where(short, -1, position)
    return npm.ffill(position)

def AGC(bp:OneDimensionalFloatArray, decay:float = 0.991) -> OneDimensionalFloatArray:
    abs_bp = abs(bp)
    peaks = np.zeros_like(bp)
    old_peak = 0
    for i, _ in enumerate(bp):
        peak = decay*old_peak
        val = abs_bp[i]
        if val > peak:
            peak = val
            
        peaks[i] = peak
        old_peak = peak
    return peaks

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
        self.bp = lambda y: ehlers.EhlersBandPass(y, self.fs, self.pc)

@dataclass
class MyStochastic(Filtering):
    window: int = 20
    upperVal: float = 0.8
    lowerVal: float = 0.2
    
    def __post_init__(self):
        self.__runMyStochastic__()
    
    def __runMyStochastic__(self):
        self.__set_filters__()
        
        filt = self.roof(self.price).output()
        
        HighestC = npm.rolling(filt, self.window, np.max)
        LowestC = npm.rolling(filt, self.window, np.min)
        
        Num = filt - LowestC
        Denom = HighestC - LowestC
        Stoc = np.divide(Num, Denom, out=np.zeros_like(filt), where = Denom!=0)
    
        self.SmoothedStoc = self.supersmoother(Stoc).output()
    
    def output(self):
        long = npm.crosses_over(self.SmoothedStoc, self.lowerVal)
        short = npm.crosses_under(self.SmoothedStoc, self.upperVal)            
        return Position(long, short)

@dataclass
class BandPassIndicator(Filtering):
    pc: int = 20
    k1: float = 0.25
    k2: float = 1.5
    bandwidth:float = 0.3

    def __post_init__(self):
        hp = ehlers.EhlersFirstOrderHP(price=self.price, pc_hp=self.pc, mult=self.k1*self.bandwidth).output()
        bp = ehlers.EhlersBandPass(price=hp, pc=self.pc, bw=self.bandwidth).output()
    
        peak = AGC(bp)
        self.signal = bp/peak
        self.trigger = ehlers.EhlersFirstOrderHP(price=self.signal, pc_hp=self.pc, mult=self.k2*self.bandwidth).output()
    
    def output(self):
        long = npm.crosses_over(self.trigger, self.signal)
        short = npm.crosses_under(self.trigger, self.signal)            
        return Position(long, short)
        
@dataclass
class EhlersModifiedRSI(Filtering):
    window: int = 20
    upperVal: float = 70
    lowerVal: float = 30
    
    def __post_init__(self):
        self.__set_filters__()

        filt = self.roof(self.price).output()
        filt_1 = npm.nanshift(filt,1)
               
        closes_up_bool = filt > filt_1
        closes_down_bool = filt < filt_1
        
        closes_up = npm.rolling(closes_up_bool, self.window, np.count_nonzero)
        closes_down = npm.rolling(closes_down_bool, self.window, np.count_nonzero)
        denom = closes_up + closes_down
        rsi = npm.fillna(100*closes_up / denom, 100)
        self.smoothed_rsi = self.supersmoother(rsi).output()

    def output(self):
        long = npm.crosses_over(self.smoothed_rsi, self.lowerVal)
        short = npm.crosses_under(self.smoothed_rsi, self.upperVal)
        position = Position(long, short)
        
        out = (40 < self.smoothed_rsi) & (self.smoothed_rsi < 60)
        position = np.where(out, 0, position)
        return position

@dataclass
class EvenBetterSinewave(Filtering):
    duration: int = 40
    
    def __post_init__(self):
        self.__set_filters__()
        hp = ehlers.EhlersFirstOrderHP(price=self.price, pc_hp=self.duration, alpha_key=2).output()
        filt = self.supersmoother(hp).output()
        filt_1 = npm.nanshift(filt, 1)
        filt_2 = npm.nanshift(filt, 2)
        
        wave = (filt + filt_1 + filt_2) / 3
        pwr = (filt**2 + filt_1**2 + filt_2**2) / 3
        self.wave = wave / np.sqrt(pwr)
    
    def output(self):
        long = self.wave > 0.9
        short = self.wave < -0.9
        position = np.where(long, 1, 0)
        position = np.where(short, -1, position)
        return position