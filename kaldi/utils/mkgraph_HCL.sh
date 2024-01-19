#!/bin/bash
# Copyright 2010-2012 Microsoft Corporation
#           2012-2013 Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script creates a fully expanded decoding graph (HCLG) that represents
# all the language-model, pronunciation dictionary (lexicon), context-dependency,
# and HMM structure in our model.  The output is a Finite State Transducer
# that has word-ids on the output, and pdf-ids on the input (these are indexes
# that resolve to Gaussian Mixture Models).
# See
#  http://kaldi.sourceforge.net/graph_recipe_test.html
# (this is compiled from this repository using Doxygen,
# the source for this part is in src/doc/graph_recipe_test.dox)

set -o pipefail

export LC_ALL=C.UTF-8

N=3
P=1
tscale=1.0
loopscale=0.1

reverse=false

for x in `seq 2`; do
  [ "$1" == "--mono" ] && N=1 && P=0 && shift;
  [ "$1" == "--quinphone" ] && N=5 && P=2 && shift;
  [ "$1" == "--reverse" ] && reverse=true && shift;
  [ "$1" == "--transition-scale" ] && tscale=$2 && shift 2;
  [ "$1" == "--self-loop-scale" ] && loopscale=$2 && shift 2;
done

if [ $# != 4 ]; then
   echo "Usage: utils/mkgraph.sh [options] <lang-dir> <model> <tree> <graphdir>"
   echo "e.g.: utils/mkgraph.sh data/lang_test train/mdl train/tree exp/tri1/graph"
   echo " Options:"
   echo " --mono          #  For monophone models."
   echo " --quinphone     #  For models with 5-phone context (3 is default)"
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

lang=$1
model=$2
tree=$3
dir=$4

mkdir -p $dir

# If $lang/tmp/LG.fst does not exist or is older than its sources, make it...
# (note: the [[ ]] brackets make the || type operators work (inside [ ], we
# would have to use -o instead),  -f means file exists, and -ot means older than).

required="$lang/L_disambig.fst $lang/G.fst $lang/phones.txt $lang/words.txt $lang/phones/silence.csl $lang/phones/disambig.int $model $tree"
for f in $required; do
  [ ! -f $f ] && echo "mkgraph.sh: expected $f to exist" && exit 1;
done

mkdir -p $lang/tmp
# Note: [[ ]] is like [ ] but enables certain extra constructs, e.g. || in
# place of -o
#if [[ ! -s $lang/tmp/LG.fst || $lang/tmp/LG.fst -ot $lang/G.fst || \
#      $lang/tmp/LG.fst -ot $lang/L_disambig.fst ]]; then
#  fsttablecompose $lang/L_disambig.fst $lang/G.fst | fstdeterminizestar --use-log=true | \
#    fstminimizeencoded | fstpushspecial | \
#    fstarcsort --sort_type=ilabel > $lang/tmp/LG.fst || exit 1;
#  fstisstochastic $lang/tmp/LG.fst || echo "[info]: LG not stochastic."
#fi
cp $lang/L_disambig.fst $lang/tmp/L.fst

cl=$lang/tmp/CL_${N}_${P}.fst

if [[ ! -s $cl || $cl -ot $lang/tmp/L.fst ]]; then
  fstcomposecontext --context-size=$N --central-position=$P \
   --read-disambig-syms=$lang/phones/disambig.int \
   --write-disambig-syms=$lang/tmp/disambig_ilabels_${N}_${P}.int \
    $lang/tmp/ilabels_${N}_${P} < $lang/tmp/L.fst | \
    fstarcsort --sort_type=ilabel > $cl
  fstisstochastic $cl  || echo "[info]: CL not stochastic."
fi

if [[ ! -s $dir/Ha.fst || $dir/Ha.fst -ot $model  \
    || $dir/Ha.fst -ot $lang/tmp/ilabels_${N}_${P} ]]; then
  if $reverse; then
    make-h-transducer --reverse=true --push_weights=true \
      --disambig-syms-out=$dir/disambig_tid.int \
      --transition-scale=$tscale $lang/tmp/ilabels_${N}_${P} $tree $model \
      > $dir/Ha.fst || exit 1;
  else
    make-h-transducer --disambig-syms-out=$dir/disambig_tid.int \
      --transition-scale=$tscale $lang/tmp/ilabels_${N}_${P} $tree $model \
       > $dir/Ha.fst || exit 1;
  fi
fi

if [[ ! -s $dir/HaCL.fst || $dir/HaCL.fst -ot $dir/Ha.fst || \
      $dir/HaCL.fst -ot $cl ]]; then
  fsttablecompose $dir/Ha.fst "$cl" | fstdeterminizestar --use-log=true \
    | fstrmsymbols $dir/disambig_tid.int | fstrmepslocal | \
     fstminimizeencoded > $dir/HaCL.fst || exit 1;
  fstisstochastic $dir/HaCL.fst || echo "HaCL is not stochastic"
fi

if [[ ! -s $dir/HCL.fst || $dir/HCL.fst -ot $dir/HaCL.fst ]]; then
  add-self-loops --self-loop-scale=$loopscale --reorder=true \
    $model < $dir/HaCL.fst |
    fstarcsort --sort_type=olabel > $dir/HCL.fst || exit 1;

  if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
    # No point doing this test if transition-scale not 1, as it is bound to fail.
    fstisstochastic $dir/HCL.fst || echo "[info]: final HCL is not stochastic."
  fi
fi

# keep a copy of the lexicon and a list of silence phones with HCLG...
# this means we can decode without reference to the $lang directory.

cp $lang/G.fst $dir/ || exit 1;
cp $lang/words.txt $dir/ || exit 1;
mkdir -p $dir/phones
cp $lang/phones/word_boundary.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
cp $lang/phones/align_lexicon.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
  # but ignore the error if it's not there.

cp $lang/phones/disambig.{txt,int} $dir/phones/ 2> /dev/null
cp $lang/phones/silence.csl $dir/phones/ || exit 1;
cp $lang/phones.txt $dir/ 2> /dev/null # ignore the error if it's not there.

# to make const fst:
# fstconvert --fst_type=const $dir/HCL.fst $dir/HCL_c.fst
am-info --print-args=false $model | grep pdfs | awk '{print $NF}' > $dir/num_pdfs
