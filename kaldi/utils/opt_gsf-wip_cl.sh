#!/bin/bash

echo -n "$1 $2 " >> RES_LM
utils/score.sh --wip $2 --lmw $1 test/graph/words.txt "ark:gzip -c -d lat.gz |" grnTruth.dat hyp 2>/dev/null |
awk 'NR==1{print $2}' | tee -a RES_LM
[ -d REP ] || mkdir -p REP; mv hyp REP/hyp-GSF${1}_WIP${2}

