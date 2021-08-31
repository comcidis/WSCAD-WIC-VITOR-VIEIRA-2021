#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import sys
import numpy as np

#------------ ARGUMENTOS QUE VÃO VIR DIRETO DA EXECUÇÃO DO PYTHON3 ----------#
numeroExemplos = int(sys.argv[1])
numeroFloat = int(sys.argv[2])
dask = sys.argv[3]
quantidadeAtributos = int(sys.argv[4])

#encoder
encoder = OneHotEncoder()

#----------------------------------------------------------------------------#
if(numeroFloat==32):
    type_dict = {'numeric1': 'float32', 'numeric2': 'float32',
             'numeric3': 'float32', 'numeric4': 'float32',
             'numeric5': 'float32', 'numeric6': 'float32',
             'numeric7': 'float32', 'numeric8': 'float32',
             'numeric9': 'float32', 'numeric10': 'float32',
             'numeric11': 'float32','numeric12': 'float32',
             'numeric13':'float32','numeric14': 'float32',
             'numeric15':'float32','numeric16': 'float32',
             'numeric17':'float32','numeric18': 'float32',
             'numeric19':'float32','numeric20': 'float32',
             'numeric21':'float32','class':'category'}
if(numeroFloat==16):
    type_dict = {'numeric1': 'float16', 'numeric2': 'float16',
             'numeric3': 'float16', 'numeric4': 'float16',
             'numeric5': 'float16', 'numeric6': 'float16',
             'numeric7': 'float16', 'numeric8': 'float16',
             'numeric9': 'float16', 'numeric10': 'float16',
             'numeric11': 'float16','numeric12': 'float16',
             'numeric13':'float16','numeric14': 'float16',
             'numeric15':'float16','numeric16': 'float16',
             'numeric17':'float16','numeric18': 'float16',
             'numeric19':'float16','numeric20': 'float16',
             'numeric21':'float16','class':'category'}
if(numeroFloat==64):
    type_dict = {'numeric1': 'float64', 'numeric2': 'float64',
             'numeric3': 'float64', 'numeric4': 'float64',
             'numeric5': 'float64', 'numeric6': 'float64',
             'numeric7': 'float64', 'numeric8': 'float64',
             'numeric9': 'float64', 'numeric10': 'float64',
             'numeric11': 'float64','numeric12': 'float64',
             'numeric13':'float64','numeric14': 'float64',
             'numeric15':'float64','numeric16': 'float64',
             'numeric17':'float64','numeric18': 'float64',
             'numeric19':'float64','numeric20': 'float64',
             'numeric21':'float64','class':'category'}
if(dask=='sim'):   
    from dask_ml.preprocessing import DummyEncoder
    from dask import dataframe as dd

    df = dd.read_csv("../../../../BaseSintetica/"+str(numeroExemplos)+"k_"+str(quantidadeAtributos)+"att_num.csv",dtype=type_dict)
    x = df.iloc[:,:-1].compute(num_workers=12)
    y = df.iloc[:,-1:].compute(num_workers=12).values.ravel()
else:
    import pandas as pd
    df = pd.read_csv("../../../../BaseSintetica/"+str(numeroExemplos)+"k_"+str(quantidadeAtributos)+"att_num.csv",dtype=type_dict)
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1:].values.ravel()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier(max_depth=10,random_state = 0, n_jobs=-1)

florestaMontada = rf.fit(x_train,y_train)

previsao = np.array(rf.predict(x_test))
acuracia = accuracy_score(y_test, previsao)

print(f'acuracia:{acuracia}')
# print(f'Tamanho x antes:{x.shape}\nTamanho x depois:{x.shape}')
