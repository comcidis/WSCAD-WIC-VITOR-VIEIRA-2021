#!/usr/bin/env python

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import sys
import numpy as np

#------------ ARGUMENTOS QUE VÃO VIR DIRETO DA EXECUÇÃO DO PYTHON3 ----------#
numeroExemplos = int(sys.argv[1])
dask = sys.argv[2]
quantidadeAtributos = sys.argv[3]

#encoder
encoder = OneHotEncoder()

#----------------------------------------------------------------------------#

type_dict = {'nominal1': 'category','nominal2': 'category',
            'nominal3': 'category', 'nominal4': 'category',
            'nominal5': 'category', 'nominal6': 'category',
            'nominal7': 'category', 'nominal8': 'category',
            'nominal9': 'category', 'nominal10': 'category',
            'nominal11': 'category','nominal12': 'category',
            'nominal13':'category','nominal14': 'category',
            'nominal15':'category','nominal16': 'category',
            'nominal17':'category','nominal18': 'category',
            'nominal19':'category','nominal20': 'category',
            'nominal21':'category','class':'category'}

if(dask=='sim'):   
    from dask_ml.preprocessing import DummyEncoder
    from dask import dataframe as dd

    df = dd.read_csv("../../../../BaseSintetica/"+str(numeroExemplos)+"k_"+str(quantidadeAtributos)+"att_categ.csv",dtype=type_dict)
   
    dfCategorico = df.loc[:, df.columns != 'class'].compute(num_workers=12)
    x = encoder.fit_transform(dfCategorico)
    y = df.iloc[:,-1:].compute(num_workers=12).values.ravel()

else:
    import pandas as pd
    df = pd.read_csv("../../../../BaseSintetica/"+str(numeroExemplos)+"k_"+str(quantidadeAtributos)+"att_categ.csv",dtype=type_dict)
    dfCategorico = df.loc[:, df.columns != 'class']
    x = encoder.fit_transform(dfCategorico)
    y = df.iloc[:,-1:].values.ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier(max_depth=10,random_state = 0, n_jobs=-1)

florestaMontada = rf.fit(x_train,y_train)

previsao = np.array(rf.predict(x_test))
acuracia = accuracy_score(y_test, previsao)

print(f'acuracia:{acuracia}')
print(f'Tamanho x antes:{dfCategorico.shape}\nTamanho x depois:{x.shape}')
