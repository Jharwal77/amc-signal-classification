import numpy as np
from scipy.signal import lfilter
CLASSES=['BPSK','QPSK','8PSK','16QAM','64QAM','GFSK','AM','FM']
def _symbols(label,n,rng):
 if label=='BPSK': return 2*rng.integers(0,2,n)-1
 if label.endswith('PSK'):
  m={'BPSK':2,'QPSK':4,'8PSK':8}[label]; return np.exp(1j*2*np.pi*rng.integers(0,m,n)/m)
 if label.endswith('QAM'):
  m=int(label[:-3]); a=np.arange(-int(np.sqrt(m))+1,int(np.sqrt(m)),2); return rng.choice(a,n)+1j*rng.choice(a,n)
 return rng.standard_normal(n)+1j*rng.standard_normal(n)
def generate(label='QPSK',num_samples=4096,snr_db=10,sample_rate=1e6,symbol_rate=1e5,cfo_hz=0,seed=0):
 rng=np.random.default_rng(seed); sps=max(2,int(sample_rate/symbol_rate)); ns=int(np.ceil(num_samples/sps)); sy=_symbols(label,ns,rng)
 base=np.repeat(sy,sps)[:num_samples].astype(complex)
 t=np.arange(num_samples)/sample_rate
 if label=='AM': base=(1+.7*np.real(base))*np.exp(1j*2*np.pi*0.08e6*t)
 elif label=='FM': base=np.exp(1j*2*np.pi*0.08e6*t+1.5*np.cumsum(np.real(base))/sample_rate)
 elif label=='GFSK': base=np.exp(1j*0.6*np.cumsum(np.real(base))/sps)
 base*=np.exp(1j*(2*np.pi*cfo_hz*t+rng.uniform(0,2*np.pi)))
 base/=np.sqrt(np.mean(np.abs(base)**2)+1e-12)
 p=10**(-snr_db/10); noise=np.sqrt(p/2)*(rng.standard_normal(num_samples)+1j*rng.standard_normal(num_samples))
 return (base+noise).astype(np.complex64), {'label':label,'snr_db':snr_db,'sample_rate':sample_rate,'symbol_rate':symbol_rate,'cfo_hz':cfo_hz}
def make_dataset(path,n_per_class=25,length=1024,seed=42):
 X=[]; y=[]; meta=[]; rng=np.random.default_rng(seed)
 for label in CLASSES:
  for _ in range(n_per_class):
   snr=float(rng.choice([-4,0,4,8,12,16])); cfo=float(rng.uniform(-3000,3000))
   x,m=generate(label,length,snr,cfo_hz=cfo,seed=int(rng.integers(1e9)))
   X.append(x);y.append(label);meta.append(m)
 np.savez_compressed(path,X=np.asarray(X),y=np.asarray(y),meta=np.asarray(meta,dtype=object))
