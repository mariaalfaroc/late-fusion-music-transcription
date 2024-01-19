#!/bin/bash

set -e
export LC_ALL=C.UTF-8

# Settings and Environment Variables
# ==================================================================================================
# --------------------------------------
# Data settings
# --------------------------------------
CHRS_LST=chars.lst                        # File containing the list of chars
TRAIN_ID_LST=ID_train.lst		  # List of line IDs of Training set
DEVEL_ID_LST=ID_valid.lst		  # List of line IDs of Validation set
GT_FILE=grnTruth.dat		          # File containing ground-truth in Kaldi format
#EXT_PM_FILE=txt				  # File extension of confidence matrix files
AMT_ConfMats="AMT_ConfMat"
AMT_Lattices="AMT/AMT_lattices/"
AMT_ConfNetworks="AMT/AMT_ConfNetworks/"
OMR_ConfMats="OMR/OMR_ConfMat"
OMR_Lattices="OMR/OMR_lattices/"
OMR_ConfNetworks="OMR/OMR_ConfNetworks/"

# --------------------------------------
# Special symbols
# --------------------------------------
BLANK_SYMB="<ctc>"                        # BLSTM non-character symbol
WHITESPACE_SYMB="barline"                  # White space symbol
DUMMY_CHAR="<DUMMY>"                      # Especial HMM used for modelling "</s>" end-sentence
# --------------------------------------
# Feature processing settings
# --------------------------------------
#LOGLKH_ALPHA_FACTOR=0.3                  # p(x|s) = P(s|x) / P(s)^LOGLKH_ALPHA_FACTOR
# --------------------------------------
# Modelling settings
# --------------------------------------
HMM_LOOP_PROB=0.5			  # Self-Loop HMM-state probability
HMM_NAC_PROB=0.5			  # BLSTM-NaC HMM-state probability
GSF=1.0		 			  # Grammar Scale Factor
WIP=-0.8				  # Word Insertion Penalty
ASF=2.5	   				  # Acoustic Scale Factor
NGRAM_ORDER=5			  # N-Gram Language Model Order
# --------------------------------------
# Decoding settings
# --------------------------------------
MAX_NUM_ACT_STATES=2007483647		  # Maximum number of active states
BEAM_SEARCH=15				  # Beam search
LATTICE_BEAM=12				  # Lattice generation beam
# --------------------------------------
# System settings
# --------------------------------------
N_CORES=12				  # Number of cores
# ==================================================================================================


# Check for installed software
# ==================================================================================================
# SRILM and KALDI Stuff
bKLDE=0
SRILM=$(which ngram-count)
[ -z "$SRILM" ] && { echo "ERROR: \"SRILM Tool\" is not installed/found "'!' 1>&2; exit 1; }
FSTOOL=$(which fstcompile)
[ -z "$FSTOOL" ] && { echo "ERROR: \"FST Tools\" are not installed/found "'!' 1>&2; exit 1; }
KALDI_A=$(which compute-wer)
[ -z "$KALDI_A" ] && { bKLDE=1; echo "ERROR Check A: Kaldi soft is not installed/found "'!' 1>&2; exit 1; }
KALDI_B=$(which feat-to-dim)
[ -z "$KALDI_B" ] && { bKLDE=1; echo "ERROR Check B: Kaldi soft is not installed/found "'!' 1>&2; exit 1; }
KALDI_C=$(which gmm-init-mono)
[ -z "$KALDI_C" ] && { bKLDE=1; echo "ERROR Check C: Kaldi soft is not installed/found "'!' 1>&2; exit 1; }
KALDI_D=$(which lattice-lmrescore)
[ -z "$KALDI_D" ] && { bKLDE=1; echo "ERROR Check D: Kaldi soft is not installed/found "'!' 1>&2; exit 1; }
KALDI_E=$(which lattice-copy)
[ -z "$KALDI_E" ] && { bKLDE=1; echo "ERROR Check E: Kaldi soft is not installed/found "'!' 1>&2; exit 1; }
KALDI_F=$(which arpa2fst)
[ -z "$KALDI_F" ] && { bKLDE=1; echo "ERROR Check F: Kaldi soft is not installed/found "'!' 1>&2; exit 1; }
KALDI_G=$(which fsttablecompose)
[ -z "$KALDI_G" ] && { bKLDE=1; echo "ERROR Check G: Kaldi soft is not installed/found "'!' 1>&2; exit 1; }

if [ $bKLDE -eq 1 ]; then
echo -e "Add to the PATH environment variable, the following line:\n"
cat << EOF >&2
KALDI_ROOT=<path-dir>/kaldi-trunk
export PATH=\$PATH:\$KALDI_ROOT/src/bin:\$KALDI_ROOT/tools/openfst/bin:\$KALDI_ROOT/src/fstbin/:\$KALDI_ROOT/src/gmmbin/:\$KALDI_ROOT/src/featbin/:\$KALDI_ROOT/src/lmbin/:\$KALDI_ROOT/src/sgmmbin/:\$KALDI_ROOT/src/sgmm2bin/:\$KALDI_ROOT/src/fgmmbin/:\$KALDI_ROOT/src/latbin/:\$KALDI_ROOT/src/nnet2bin/:\$KALDI_ROOT/src/nnetbin:\$KALDI_ROOT/src/online2bin/:\$KALDI_ROOT/src/ivectorbin/
export PATH
EOF
fi


# ==================================================================================================

# Check required "utils" directory with scripts
# ==================================================================================================
[ -d utils ] || { echo "ERROR: \"utils\" directory does not exist "'!' 1>&2; exit 1; }
export PATH="$(pwd)/utils:$PATH"
# ==================================================================================================

# Splitting reference transcripts according to the train and devel ID lists
############################################################################################################
[ -f data/train/text ] ||
{
#  echo "Processing training transcripts into Kaldi format ..." 1>&2
#  prepare_transc_cl.sh ${TRAIN_GT_DIR} ${EXT_GT_FILE} data/train PROC/ID-train.lst
  [ -d data/train/ ] || mkdir -p data/train/
  awk -v idf="$TRAIN_ID_LST" 'BEGIN{while (getline < idf > 0) IDs[$1]=""}
                              {if ($1 in IDs) print}' $GT_FILE > data/train/text
}
[ -f data/test/text ] ||
{
#  echo "Processing development transcripts into Kaldi format ..." 1>&2
#  prepare_transc_cl.sh ${DEVEL_GT_DIR} ${EXT_GT_FILE} data/test PROC/ID-devel.lst
  [ -d data/test/ ] || mkdir -p data/test/
  awk -v idf="$DEVEL_ID_LST" 'BEGIN{while (getline < idf > 0) IDs[$1]=""}
                              {if ($1 in IDs) print}' $GT_FILE > data/test/text
}
############################################################################################################




# Prepare Kaldi's lang directories
############################################################################################################
# Preparing Lexic (L)
[ -d data/train/lang ] ||
{
  echo "Generating lexic model ..." 1>&2
  prepare_lang_cl-ds.sh data/train ${CHRS_LST} "${BLANK_SYMB}" "${WHITESPACE_SYMB}" "${DUMMY_CHAR}"
}

# Preparing LM (G)
[ -f data/train/lang_test/G.fst ] ||
{
  echo "Generating ${NGRAM_ORDER} character-level language model ..." 1>&2
  [ -f data/train/lang/LM.arpa ] || {
    cat data/train/text | cut -d " " -f 2- |
    ngram-count -text - -lm data/train/lang/LM.arpa -order ${NGRAM_ORDER} -kndiscount2 -kndiscount3 -kndiscount4 -kndiscount5 -kndiscount6 -interpolate; }
#    ngram-count -text - -lm data/train/lang/LM.arpa -order ${NGRAM_ORDER} -kndiscount -interpolate; }
  # ngram-count -text - -lm data/train/lang/LM.arpa -order ${NGRAM_ORDER} -kndiscount2 -interpolate; }
  prepare_lang_test-ds.sh data/train/lang/LM.arpa data/train/lang data/train/lang_test "$DUMMY_CHAR"
}
############################################################################################################


# Prepare HMM models
############################################################################################################
# Create HMM topology file
[ -d train ] ||
{
  echo "Creating character HMM topologies ..." 1>&2
  mkdir train
  phones_list=( $(cat data/train/lang_test/phones/{,non}silence.int) )
  featdim=$(cat chars.lst|wc -l)
  dummyID=$(awk -v d="$DUMMY_CHAR" '{if (d==$1) print $2}' data/train/lang/phones.txt)
  blankID=$(awk -v bs="${BLANK_SYMB}" '{if (bs==$1) print $2}' data/train/lang/pdf_blank.txt)
  create_proto_rnn-ds.sh $featdim ${HMM_LOOP_PROB} ${HMM_NAC_PROB} train ${dummyID} ${blankID} ${phones_list[@]}
}
############################################################################################################



# Compose FSTs
############################################################################################################
[ -d test/ ] ||
{
  touch data/train/lang_test/phones/silence.csl  # for the moment
  echo "Creating global SFS automaton for decoding ..." 1>&2
  mkdir test
  mkgraph.sh --mono --transition-scale 1.0 --self-loop-scale 1.0 \
                   data/train/lang_test train/new.mdl train/new.tree test/graph
}
############################################################################################################



for u in ${AMT_ConfMats}*; do


# Lattice Generation
############################################################################################################

  echo "Generating lattices for $(basename $u )" 1>&2
  copy-matrix "ark,t:${AMT_ConfMats}$(basename $u )" "ark,scp:$(basename $u ).ark,$(basename $u ).scp"
  split -d -n l/${N_CORES} -a 3 $(basename $u ).scp part-
  mkdir lattices
  for n in $(seq -f "%03.0f" 0 1 $[N_CORES-1]); do
    echo "launching subprocess in core $n ..." 1>&2
    latgen-faster-mapped --verbose=2 --allow-partial=true --acoustic-scale=${ASF} --max-active=${MAX_NUM_ACT_STATES} \
                         --beam=${BEAM_SEARCH} --lattice-beam=${LATTICE_BEAM} train/new.mdl \
			 test/graph/HCLG.fst scp:part-$n "ark:|gzip -c > lattices/lat_$n.gz" \
    		         ark,t:lattices/RES_$n 2>lattices/LOG-Lats-$n &
  done
  echo "Waiting for finalization of the ${N_CORES} subprocesses ..." 1>&2
  wait
  lattice-copy "ark:gunzip -c lattices/lat_*.gz |" "ark:|gzip -c > ${AMT_Lattices}$(basename $u ).gz"
  lattice-mbr-decode --lm-scale=0 "ark:zcat ${AMT_Lattices}$(basename $u ).gz|" ark:/dev/null ark:/dev/null ark,t:${AMT_ConfNetworks}$(basename $u )
  rm -rf lattices/ part-* $(basename $u ) *.scp *.ark

done
#########################################################################################################



for u in ${OMR_ConfMats}*; do

# Lattice Generation
############################################################################################################

 echo "Generating lattices for $(basename $u )" 1>&2
 copy-matrix "ark,t:${OMR_ConfMats}$(basename $u )" "ark,scp:$(basename $u ).ark,$(basename $u ).scp"
 split -d -n l/${N_CORES} -a 3 $(basename $u ).scp part-
 mkdir lattices
 for n in $(seq -f "%03.0f" 0 1 $[N_CORES-1]); do
   echo "launching subprocess in core $n ..." 1>&2
   latgen-faster-mapped --verbose=2 --allow-partial=true --acoustic-scale=${ASF} --max-active=${MAX_NUM_ACT_STATES} \
                        --beam=${BEAM_SEARCH} --lattice-beam=${LATTICE_BEAM} train/new.mdl \
			 test/graph/HCLG.fst scp:part-$n "ark:|gzip -c > lattices/lat_$n.gz" \
   		         ark,t:lattices/RES_$n 2>lattices/LOG-Lats-$n &
 done
 echo "Waiting for finalization of the ${N_CORES} subprocesses ..." 1>&2
 wait
 lattice-copy "ark:gunzip -c lattices/lat_*.gz |" "ark:|gzip -c > ${OMR_Lattices}$(basename $u ).gz"
 lattice-mbr-decode --lm-scale=1  "ark:zcat ${OMR_Lattices}$(basename $u ).gz|" ark:/dev/null ark:/dev/null ark,t:${OMR_ConfNetworks}$(basename $u )
 rm -rf lattices/ part-* $(basename $u ) *.scp *.ark

done
############################################################################################################




exit 0
