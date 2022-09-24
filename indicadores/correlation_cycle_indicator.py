import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from numpy_methods import nanshift, ffill, crosses_over, crosses_under
from dataclasses import dataclass
from nptyping import NDArray, Shape, Float

OneDimensionalFloatArray = NDArray[Shape['Any'], Float]

@dataclass
class CorrelationIndicator:
    price: OneDimensionalFloatArray
    # date: pd.DataFrame
    bars: int = 48
    period: int = 20
    
    def __post_init__(self):
        window_range = np.arange(self.bars)
        arg = 2*np.pi*window_range / self.period
        self.cosine = np.cos(arg)
        self.negative_sine = -np.sin(arg)
        self._getPhase()

    def _advancedPhase(self, phase):
        lagged_phase = nanshift(phase,1)
        sine = np.sin(phase)
        cosine = np.cos(phase)
        lagged_cosine = np.cos(lagged_phase)
        
        right_semicircle = cosine > 0
        left_semicircle = cosine < 0
        
        upper_semicircle = sine > 0
        lower_semicircle = sine < 0
        
        first_quarter = upper_semicircle & right_semicircle
        second_quarter =  upper_semicircle & left_semicircle
        third_quarter =  lower_semicircle & left_semicircle
        fourth_quarter =  lower_semicircle & right_semicircle

        lagged_upper_semicircle = nanshift(upper_semicircle,1)
        lagged_lower_semicircle = nanshift(lower_semicircle,1)
        lagged_second_quarter = nanshift(second_quarter,1)
        lagged_fourth_quarter = nanshift(fourth_quarter,1)
    
        advanced_a = upper_semicircle * lagged_upper_semicircle * (cosine < lagged_cosine)
        advanced_b = lower_semicircle * lagged_lower_semicircle * (cosine > lagged_cosine)
        advanced_c = first_quarter * lagged_fourth_quarter
        advanced_d = third_quarter * lagged_second_quarter
        
        return advanced_a + advanced_b + advanced_c + advanced_d
        
    def _getPhase(self):
        autocorrelation_real = list()
        autocorrelation_imag = list()
        price_last_is_first = self.price[::-1]
        for lag in range(self.price.size - self.bars):
            autocorrelation_real = [ np.corrcoef(price_last_is_first[lag:lag + self.bars], self.cosine)[0,1] ] + autocorrelation_real
            autocorrelation_imag = [ np.corrcoef(price_last_is_first[lag:lag + self.bars], self.negative_sine)[0,1] ] + autocorrelation_imag
                  
        self.df = pd.DataFrame({'Price': self.price[self.bars:]})
        self.df["Phase"] = np.arctan2(autocorrelation_real, autocorrelation_imag).flatten()
        
        while True:
            temp_phase = np.where(self._advancedPhase(self.df["Phase"]) == False, np.nan, self.df["Phase"])
            temp_phase = ffill(temp_phase)
            if (temp_phase == self.df["Phase"]).all():
                break
            else:
                self.df["Phase"] = temp_phase

        self.df["Lead Phase"] = self.df["Phase"] + np.pi/2
        
        self.df["Regime"] = np.where( crosses_over(autocorrelation_imag, 0), 1, np.nan)
        self.df["Regime"] = np.where( crosses_under(autocorrelation_imag, 0), -1, self.df["Regime"])
        self.df["Regime"] = self.df["Regime"].ffill().fillna(0)
        self.df["Regime"] = np.where(self.df["Phase"] == self.df["Phase"].shift(1), 0, self.df["Regime"])
        
        self.df["Returns"] = self.df["Price"] / self.df["Price"].shift(1)
        self.df["Cum Returns"] = self.df["Returns"].cumprod()
        self.df["Strategy"] = self.df["Returns"]**self.df["Regime"].shift(1)
        
        self.df["Equity"] = self.df["Strategy"].cumprod().fillna(1)
        # self.df.index = self.date[self.bars:]
        
        fig, axes = plt.subplots(2, 1,figsize=(16,9), sharex=True, gridspec_kw={'height_ratios': [5, 1]})
        axes[0].set_title("Estrategia sobre el NASDAQ100 con el indicador de correlación")
        axes[0].plot(self.df["Cum Returns"])
        axes[0].plot(self.df["Equity"])
        axes[0].legend(["Cum Returns", "Equity"], loc="upper left")
        axes[0].grid()
        
        axes[1].set_title("Phase")
        axes[1].plot(self.df["Phase"], color="green")
        axes[1].set_yticks([-np.pi, np.pi])
        pi = unicodedata.lookup("GREEK SMALL LETTER PI")
        axes[1].set_ylim([-3.5, 3.5])
        axes[1].set_yticklabels([f"-{pi}", pi])
        axes[1].grid()