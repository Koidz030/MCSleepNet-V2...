# coding: UTF-8
##############################
### translate_rawdata_to_csv.py
### 2015 Decmber 
### Editted time : 2016
### Author : Yuta Suzuki, (Leo Ota revised)
###
### Data collocted by IIIS
### 
### Data/3eg/mouse_type".3eg 		<--- 3eg
### >
### Data/raw/Date_"mouse_type".csv 	<--- 3eg
### Data/raw/eeg_"mouse_type".csv 	<--- 3eg
### Data/raw/emg_"mouse_type".csv 	<--- 3eg
### Data/raw/epoch_"mouse_type".csv 	<--- 3eg
##############################

#from progressbar import Percentage, ProgressBar, Bar, ETA	# progress bar 
import csv		# csv�
import struct												# 
from scipy.io import loadmat
import os
import glob
import pandas as pd

# mode_mat = "slpy"
mode_mat = "slpy2"

# 3eg
#files = glob.glob("/data2/clearData20190330/3eg/*.3eg")
#df_mouselist = pd.read_csv('/home/ota/analysis_clearData/dataset_20190405_clearMouse3.csv')
#df_mouselist = pd.read_csv('/home/ota/analysis_clearData/dataset_20190405_clearMouse4.csv')
df_mouselist = pd.read_csv('/data2/clearData20190330/Y0mouselist.csv')

# for filepath in files:
for dataname in df_mouselist['name']:#num in np.arange(1,len(files)+1):
	# 
	#filepath = glob.glob("/data2/clearData20190330/3eg/*_%04s_*.3eg" % dataname)[0]
	filepath = glob.glob("/data2/clearData20190330/3eg/*_%12s_*.3eg" % dataname)[0]
	# 
	mousename = dataname#"%04d" % num�
	#filename, ext = os.path.splitext(os.path.basename(filepath))
	#mat_file = glob.glob("/data2/clearData20190330/mat/%04s_*.mat" % dataname)[0]
	mat_file = glob.glob("/data2/clearData20190330/mat/%12s_*.mat" % dataname)[0]
	dataset = loadmat(mat_file)
	if mode_mat == "slpy2":
		epoch_start = dataset["AnalyzedEpoch_Basal"][0][0] * 5
		epoch_end = dataset["AnalyzedEpoch_Basal"][0][1] * 5
		epoch_num = epoch_end - epoch_start + 5
		date_start = dataset["Selected_Date_Basal"][0][0][0]
		date_end = dataset["Selected_Date_Basal"][-1][0][0]
	elif mode_mat == "slpy":
		epoch_num = dataset["EpochList_Basal"][-1][1]
		date_start = dataset["Selected_Date_Basal"][0][0][0]

	print "file(3eg): %s" % filepath
	print "file(mat): %s" % mat_file
	print "mouse: %s" % mousename
	if mode_mat == "slpy2":
		print "StartEpoch#: ", epoch_start
		print "EndEpoch#: ", epoch_end
	print "epoch num: %d" % epoch_num
	print "Duration(start): ", date_start
	print "Duration(end): ", date_end
	epoch_count = 0
	
	# 3eg
	f = open(filepath, "rb")
	
	# slpy2 file can overwrite t mat file extention. 
	if mode_mat == "slpy2":
		f.read(4008 * (epoch_start - 5) )
	elif mode_mat == "slpy":
		while True:
			buf = f.read(8)
			date_tmp = struct.unpack("d", buf)
			f.read(4000)
			if date_tmp[0] > date_start:
				break

	# 
	csv_date = open("/data2/clearData20190330/raw/date_%s.csv" % mousename, "wb")
	csv_eeg = open("/data2/clearData20190330/raw/eeg_%s.csv" % mousename, "wb")
	csv_emg = open("/data2/clearData20190330/raw/emg_%s.csv" % mousename, "wb")
	csv_epoch = open("/data2/clearData20190330/raw/epoch_%s.csv" % mousename, "wb")
	writer_date = csv.writer(csv_date)
	writer_eeg = csv.writer(csv_eeg)
	writer_emg = csv.writer(csv_emg)
	writer_epoch = csv.writer(csv_epoch)

	# 
	if mode_mat == "slpy2":
		writer_epoch.writerow(["start", "end", "num.4s", "num.20s"])
		writer_epoch.writerow([epoch_start, epoch_end, epoch_num, (epoch_num / 5)])
	elif mode_mat == "slpy":
		writer_epoch.writerow(["start", "num.4s", "num.20s"])
		writer_epoch.writerow([date_start, epoch_num, (epoch_num / 5)])
	csv_epoch.close()
	
	# 
	#pb = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()], maxval=epoch_num).start()
	
	# 
	while epoch_count < epoch_num:
		# Date Time (double:8byte) (format: YYYYMMDDhhmmss)
		buf = f.read(8)
		date_tmp = struct.unpack("d", buf)
		for date in date_tmp:
			writer_date.writerow([date])
		
		# EEG(int16:2byte) 1000 byte
		buf = f.read(2000)
		eeg_tmp = struct.unpack("h"*1000, buf)
		for eeg in eeg_tmp:
			writer_eeg.writerow([eeg])
		
		# EMG(int16:2byte) 1000 byte
		buf = f.read(2000)
		emg_tmp = struct.unpack("h"*1000, buf)
		for emg in emg_tmp:
			writer_emg.writerow([emg])
		
		# 
		epoch_count += 1
		
		#pb.update(epoch_count)
	
	# 
	f.close()
	#pb.finish()
	
	# CSV
	csv_date.close()
	csv_eeg.close()
	csv_emg.close()
	csv_epoch.close()
