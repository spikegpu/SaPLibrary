import os
import sys

MATRIX_PATH = '/home/ali/CUDA_project/reordering/matrices'
EXEC_PATH = '/home/ali/CUDA_project/SpikeLibrary/SpikeLibrary/examples/RELEASE/test'
EXEC_NAME = 'driver_test'
SAFE_FACT = ''
FILE_LIST_PATH = '/home/ali/CUDA_project/SpikeLibrary/SpikeLibrary/examples/test'
#SAFE_FACT = '--safe-fact'


length = len(sys.argv)
if length > 1:
	EXEC_PATH = sys.argv[1]

if length > 2:
	MATRIX_PATH = sys.argv[2]

if length > 3:
	FILE_LIST_PATH = sys.argv[3]

f = open('{0}/file_list.txt'.format(FILE_LIST_PATH), 'r')
status=0

mat_name = ''
num_part = ''


for line in f:
	if status == 0:
		mat_name = line
		status = 1
	else:
		num_part = line
		os.system('{5}/{0} -p={1} -m={2}/{3} {4}'.format(EXEC_NAME, num_part.strip(), MATRIX_PATH, mat_name.strip(), SAFE_FACT, EXEC_PATH))
		status = 0
