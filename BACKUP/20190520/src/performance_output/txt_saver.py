"""
Created on Sun May 19 16:56:05 2019

@author: Kiprot
"""
import numpy as np
def PerformanceSaver(data_obj, run_data, n_predict_once, num_unrollings, batch_size):
	''' 
	This is used to return the lowest MSE run and a text file
	with the relevant time sequence parameters:
	1) train data sequence length (done by remove_data setting)
	2) num_unrolling (how many time steps the model looks into)
	3) batch_size (number of data samples at each time step)
	4) n_predict_once (how many steps are predicted for in the future)
	and the MSE per time step in a text file
	to keep track of how changes to the time sequence length 
	affects the performance of the model
	'''
	SAVE_FOLDER = 'src/performance_output/PerformanceFiles/' # Folder to save the run results
	title = SAVE_FOLDER + 'Ntot' + str(np.shape(data_obj.train_data)[0]) + '_Npred' + str(n_predict_once)+'.txt' # Title for text file
	header = np.array(('Train data size =' + str(np.shape(data_obj.train_data)[0]), 
					   'Num_unrolling = ' + str(num_unrollings),
					   'batch_size = ' + str(batch_size),
					   'n_predict_once = ' + str(n_predict_once)))
	header = np.reshape(header, (4,1))
	
	run_data = run_data[1:]  # Removing an empty placeholder in the run_data
	
	# Finding the run with the lowest MSE
	for i in range(len(run_data)):
		dat = run_data[i][0]
		MSE = dat[-7:-1]
		lowest_temp = [i, MSE]
		if i==0:
			lowest_perm = lowest_temp
		else:
			if lowest_perm[1]>lowest_temp[1]:
				lowest_perm = lowest_temp
		
	# Saving best prediction to text file
	best_pred = np.array('Best prediction epoch: ' +str(lowest_perm[0]+1) + ' with MSE ' +str(lowest_perm[1]))
	
	# Collecting the text file together
	output_file = np.vstack((header,best_pred,run_data)) 
	
	# Outputting text file
	np.savetxt(title, output_file, fmt = '%s')
	print('Best prediction epoch: ', lowest_perm[0]+1, ' with MSE ', lowest_perm[1])
	
	return lowest_perm[0] # Return best prediction epoch