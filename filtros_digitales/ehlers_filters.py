import numpy as np
import scipy.signal
from dataclasses import dataclass, field
from nptyping import NDArray, Shape, Float

OneDimensionalFloatArray = NDArray[Shape['Any'], Float]
f = lambda x: field(default=x, init=False, repr=False)

@dataclass
class FilterOutput:
    price: OneDimensionalFloatArray
    fs: int = 1
    
    def Filter(self) -> OneDimensionalFloatArray:
        zi = scipy.signal.lfilter_zi(self.NumeratorZcoefs, self.DenominatorZcoefs)
        return lambda series: scipy.signal.lfilter(self.NumeratorZcoefs,
                                                   self.DenominatorZcoefs,
                                                   series,
                                                   zi=zi*series[0])

@dataclass
class LowPassFilter(FilterOutput):
    pc_lp: int = 10
    
    def __get_lp_filters__(self):
        self.lp_in_series = self.Filter()
        self.lp_single_output, _ = self.lp_in_series(self.price)
    
@dataclass
class HighPassFilter(FilterOutput):
    pc_hp: int = 48
    
    def __get_alpha__(self) -> float:
        pi: float = np.pi
        period: float = self.fs*self.pc_hp
        arg: float = self.mult*2*pi/period
        
        alpha1: float = (np.cos(arg) + np.sin(arg) - 1) / np.cos(arg)
        alpha2: float = (1 - np.sin(2*pi/period)) / np.cos(2*pi/period)
        
        # Pick Alpha 1 or 2
        alphas: dict[int, float] = {1: alpha1, 2: alpha2}
        return alphas[self.alpha_key]

    def __get_hp_filters__(self):
        self.hp_in_series = self.Filter()
        self.hp_single_output, _ = self.hp_in_series(self.price)

@dataclass
class EhlersFirstOrderHP(HighPassFilter):
    alpha_key: int = 1
    mult: int = f(1)
    
    def __post_init__(self):
        alpha: float = self.__get_alpha__()
        self.NumeratorZcoefs: list[float] = [(1 + alpha), -(1 + alpha)]
        self.DenominatorZcoefs: list[float] = [2, -2*(1 - alpha)]

    def output(self):
        return self.hp_single_output

@dataclass   
class EhlersSecondOrderHP(HighPassFilter):
    alpha_key: int = f(1)
    mult: float = np.sqrt(2)/2
    
    def __post_init__(self):
        self.__runEhlersSecondOrderHP__()

    def __runEhlersSecondOrderHP__(self):
        alpha: float = self.__get_alpha__()
        c2: float = 1 - alpha
        c3: float = (1 - alpha)**2
        c1: float = (1 - alpha / 2)**2
        
        self.NumeratorZcoefs: list[float] = [c1, -2*c1, c1]
        self.DenominatorZcoefs: list[float] = [1, -2*c2, c3]
        self.__get_hp_filters__()

    def output(self):
        return self.hp_single_output
        
@dataclass
class SuperSmoother(LowPassFilter):
    def __post_init__(self):
        self.__runSuperSmoother__()
    
    def __runSuperSmoother__(self):
        period: float = self.fs * self.pc_lp
        
        a1: float = np.exp(-np.sqrt(2)*np.pi/period)
        b1: float = 2*a1*np.cos(np.sqrt(2)*np.pi/period)
        c2: float = b1
        c3: float = -a1*a1
        c1: float = 1 - c2 - c3
    
        self.NumeratorZcoefs:list[float] = [c1, c1]
        self.DenominatorZcoefs:list[float] = [2, -2*c2, -2*c3]
        self.__get_lp_filters__()
        
    def output(self):
        return self.lp_single_output
        
@dataclass
class EhlersRoofingFilter(EhlersSecondOrderHP, SuperSmoother):
    def __post_init__(self):
        self.__runEhlersSecondOrderHP__()
        self.__runSuperSmoother__()
        self.roof_output, _ = self.lp_in_series(self.hp_single_output)
    
    def output(self):
        return self.roof_output