import os
import argparse
import numpy as np
from itertools import groupby
import shutil

def process_inf(line):
	line_split = np.array(line.split())
	infs = np.where(line_split == '-inf')[0]
	line_float = [float(u) for u in line_split]
	
	for idx in infs:
		line_float[idx] = np.log(0.00000000001)

	return " ".join([str(u) for u in line_float]) + '\n'



def read_ConfMat(args):

	# Reading vocab:
	with open(os.path.join(os.path.split(args.src)[0], 'chars.lst')) as f:
		FileRead = f.readlines()

	w2i = {FileRead[i].strip():i for i in range(len(FileRead))}
	i2w = {i:FileRead[i].strip() for i in range(len(FileRead))}



	# Creating destination folders:
	if not os.path.exists(args.dst): os.makedirs(args.dst)
	if not os.path.exists(os.path.join(args.greedy, 'Results')): os.makedirs(os.path.join(args.greedy, 'Results'))


	fgreedy = open(os.path.join(args.greedy, 'Results', 'greedy.txt'), 'w')


	greedy = list()
	# Processing file:
	with open(args.src) as infile:
		for line in infile:
			if '[' in line:
				fout = open(os.path.join(args.dst, line.split()[0]), 'w')
				fgreedy.write(line.split()[0] + ' ')
			if 'inf' in line: line = process_inf(line)
			fout.write(line)
			if '[' not in line and ']' not in line: greedy.append(np.argmax([float(u) for u in line.split()]))
			if ']' in line:
				greedy = [i2w[i[0]] for i in groupby(greedy) if i[0] != len(w2i)]
				fgreedy.write(" ".join(greedy) + '\n')
				greedy = list()

				fout.close()
	fgreedy.close()
	return





if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Adapting data for Kaldi.')
	parser.add_argument('--src', type=str, required = True, help='Input file')
	parser.add_argument('--dst', type=str, required = True, help='Destination path for the files')
	parser.add_argument('--greedy', type=str, required = False, help='Destination path for the greedy decoding')



	args = parser.parse_args()
	# Obtaining confusion matrices and greedy decoding from posteriograms:
	read_ConfMat(args)

	# Copying GT data
	if not os.path.exists(os.path.join(os.path.split(args.greedy)[0], 'grnTruth.dat')):
		shutil.copy(os.path.join(os.path.split(args.src)[0], 'grnTruth.dat'), os.path.join(os.path.split(args.greedy)[0], 'grnTruth.dat'))

	print("hello")


