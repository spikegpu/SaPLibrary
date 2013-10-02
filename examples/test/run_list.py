import os

MATRIX_PATH = '/home/ali/CUDA_project/reordering/matrices/'
EXEC_NAME = 'driver_test'
SAFE_FACT = ''
#SAFE_FACT = '--safe-fact'

f = open('file_list.txt', 'r')
status=0

mat_name = ''
num_part = ''

for line in f:
	if status == 0:
		mat_name = line
		status = 1
	else:
		num_part = line
		os.system('./{0} -p={1} -m={2}{3} {4}'.format(EXEC_NAME, num_part.strip(), MATRIX_PATH, mat_name.strip(), SAFE_FACT))
		status = 0
