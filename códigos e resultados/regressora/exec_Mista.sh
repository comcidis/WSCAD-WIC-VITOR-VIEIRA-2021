#!/bin/bash

EXEMPLO=("" "5000")
FLOAT=("" "64" "32" "16")
DASK=("" "sim" "nao")
ATRIBUTOS=("" "21")

for i in $(seq 10)
do
for j in $(seq 1)
do
for l in $(seq 3)
do
for k in $(seq 2)
do
for m in $(seq 1)
do
        EXECUCAOPERF="sudo time perf stat -x | -e /power/energy-pkg/,/power/energy-cores/,/power/energy-gpu/"
        COMANDO2="sudo perf stat -x | -e instructions,cycles,cpu-clock,cpu-migrations,branches,branch-misses,context-switches,bus-cycles,cache-references,cache-misses,mem-loads,mem-stores,L1-dcache-stores,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-store-misses,LLC-stores,LLC-store-misses,LLC-loads,LLC-load-misses,minor-faults,page-faults"
        APP="python3.8 floresta_Reg_Mista.py ${EXEMPLO[j]} ${FLOAT[l]} ${DASK[k]} ${ATRIBUTOS[m]}"
        $EXECUCAOPERF $COMANDO2 $APP &>Resultado_$[i]_${EXEMPLO[j]}kExemplos_FLOAT${FLOAT[l]}_DASK_${DASK[k]}_ATRIBUTOS_${ATRIBUTOS[m]}.txt
if [ -d "Resultados/${EXEMPLO[j]}kMista" ]
then
        mv Resultado_$[i]_${EXEMPLO[j]}kExemplos_FLOAT${FLOAT[l]}_DASK_${DASK[k]}_ATRIBUTOS_${ATRIBUTOS[m]}.txt Resultados/${EXEMPLO[j]}kMista
else
        mkdir -p Resultados/${EXEMPLO[j]}kMista
        mv Resultado_$[i]_${EXEMPLO[j]}kExemplos_FLOAT${FLOAT[l]}_DASK_${DASK[k]}_ATRIBUTOS_${ATRIBUTOS[m]}.txt Resultados/${EXEMPLO[j]}kMista
fi
done
done
done
done
done