import numpy as np
def normalize_iq(x):
 x=np.asarray(x,dtype=np.complex64); x=x-np.mean(x); return x/np.sqrt(np.mean(np.abs(x)**2)+1e-12)
def to_features(x,nfft=256):
 x=normalize_iq(x); i=np.real(x); q=np.imag(x); amp=np.abs(x); ph=np.unwrap(np.angle(x)); dph=np.diff(ph)
 f=np.fft.fftfreq(nfft); p=np.abs(np.fft.fft(x[:nfft]*np.hanning(min(len(x),nfft)),n=nfft))**2
 return np.array([i.mean(),q.mean(),i.std(),q.std(),amp.mean(),amp.std(),np.mean(dph),np.std(dph),np.max(p),np.argmax(p)/nfft,*np.quantile(amp,[.1,.25,.5,.75,.9])],dtype=np.float32)
