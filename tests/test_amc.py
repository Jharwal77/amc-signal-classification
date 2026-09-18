import sys, numpy as np
sys.path.insert(0,'src')
from amc.signal_generator import generate
from amc.preprocessing import normalize_iq,to_features
from amc.estimators import estimate_parameters
def test_generation():
 x,m=generate('QPSK',512,10,seed=1); assert x.shape==(512,) and np.iscomplexobj(x)
def test_normalization():
 x=normalize_iq(np.ones(100)+1j*np.arange(100)); assert abs(np.mean(x))<1e-5 and abs(np.mean(abs(x)**2)-1)<1e-5
def test_features_and_estimator():
 x,_=generate('BPSK',512,10,seed=2); assert to_features(x).shape[0]>=10; assert 'occupied_bandwidth_hz' in estimate_parameters(x)
