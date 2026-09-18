import numpy as np
from scipy.signal import welch
def estimate_parameters(x,fs=1e6):
 x=np.asarray(x); f,p=welch(x,fs=fs,nperseg=min(256,len(x)),return_onesided=False); p=np.fft.fftshift(p); f=np.fft.fftshift(f); p=np.maximum(p,1e-15); total=np.sum(p); c=float(np.sum(f*p)/total); cdf=np.cumsum(p)/total; lo=float(f[np.searchsorted(cdf,.005)]); hi=float(f[np.searchsorted(cdf,.995)])
 floor=float(np.percentile(p,20)); sig=float(max(np.mean(p)-floor,1e-12)); snr=float(10*np.log10(sig/floor));
 return {'snr_db':snr,'center_frequency_hz':c,'occupied_bandwidth_hz':hi-lo,'cfo_hz':c,'symbol_rate_baud':None}
