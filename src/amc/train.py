import argparse, json, os, joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import to_features
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/demo.npz');ap.add_argument('--out',default='models/rf.joblib');args=ap.parse_args()
 d=np.load(args.data,allow_pickle=True); X=np.array([to_features(x) for x in d['X']]); y=d['y'].astype(str)
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
 model=RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=42,n_jobs=-1);model.fit(Xtr,ytr)
 print(classification_report(yte,model.predict(Xte)));os.makedirs(os.path.dirname(args.out),exist_ok=True);joblib.dump(model,args.out)
 with open(args.out+'.json','w') as f: json.dump({'features':'iq_statistics_v1','classes':sorted(set(y))},f)
if __name__=='__main__': main()
