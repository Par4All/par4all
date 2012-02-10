#! /usr/bin/python

# -*- coding: utf-8 -*-

"""
Validation utility for Par4All

Add object oriented organization above PIPS validation.

Introduce the concept of validation class.
"""
import os, string, re, optparse, glob, shutil, stat, multiprocessing, subprocess, datetime, signal

# Init variables
p4a_root = ''
par4ll_validation_dir = ''
log_file_path = ''

# Define directories for validation, path for log file, and check P4A_ROOT exists.
for root, subfolders, files in os.walk(os.getcwd()):
	if os.path.exists(root+'/p4a_validate_class.py'):
		p4a_root = root
		par4ll_validation_dir = str(p4a_root)+'/../../packages/PIPS/validation/'

		# path for log file
		log_file_path = root


extension = ['.c','.F','.f','.f90','.f95']
script = ['.tpips','.tpips2','.test','.py']

# warning and failed status
warning = ['skipped','timeout','orphan']
failed = ['changed','failed']

# get default architecture and tpips/pips
arch = ''
for root, subfolders, files in os.walk(par4ll_validation_dir+'/..'):
	if os.path.exists(root+'/arch.sh'):
		arch = subprocess.Popen([root+'/arch.sh'], stdout=subprocess.PIPE).communicate()[0]

# Timeout
timeout = 600 # Time in second
timeout_value = 203 # Value of timeout status

# Orphan status
orphan_status = 202

# Multiprocessing
nb_cpu = multiprocessing.cpu_count()

# List of directories that will be tested
dir_list = list()

# Alarm class (timeout)
class Alarm(Exception):
  pass

### Raise Alarm ###
  def alarm_handler(self,signum,frame):
	raise Alarm

# Functions ##################
### Write in log ###
def write_log(status,log_file,test_name_path):
	# Lock to have only one process who write to log_file
	lock = multiprocessing.Lock()
	lock.acquire()
	# Write status
	file_result = open(log_file,'a')
	file_result.write ('%s: %s\n' % (status,test_name_path.replace(par4ll_validation_dir,'')))
	file_result.close()
	# Unlock
	lock.release()

### run test ###
def process_timeout(command,status,queue_err,queue_output,shell_value):

	process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=shell_value)

	# Alarm for timeout
	signal.signal(signal.SIGALRM, Alarm().alarm_handler)
	signal.alarm(timeout)
	communicate=['','']
	try:
		communicate = process.communicate()
		signal.alarm(0)  # reset the alarm
	except Alarm:
		# kill process after timeout
		process.kill()
		status.value = timeout_value

	# Output of the test
	queue_output.put(communicate[0])

	# Error of the test
	queue_err.put(communicate[1])

	if status.value != timeout_value:
		# Status
		status.value = process.returncode

###### Function that kills a process after a timeout ######
def run_process(command,shell_value):
	status = multiprocessing.Value('i',0)
	output = ''
	err = ''

	# Queues
	queue_output = multiprocessing.Queue()
	queue_err = multiprocessing.Queue()

	# Create and run process
	process_timeout(command,status,queue_err,queue_output,shell_value)

	# Output and error
	output = queue_output.get()
	err = queue_err.get()

	queue_output.close()
	queue_err.close()

	return status.value,output,err

#### Command to test ###
def command_test(directory_test_path,test_name_path,err_file_path,test_file_path):
	# tpips2 has a different output
	tpips2 = False
	if (os.path.isfile(test_name_path+".test")):
		command = [test_name_path+".test",""]
		(int_status,output,err) = run_process(command,True)

	elif (os.path.isfile(test_name_path+".tpips")):
		command = ["tpips",test_name_path+".tpips"]
		(int_status,output,err) = run_process(command,False)

	elif (os.path.isfile(test_name_path+".tpips2")):
		command = ["tpips",test_name_path+".tpips2"]
		tpips2 = True
		(int_status,output,err) = run_process(command,False)

	elif (os.path.isfile(test_name_path+".py")):
		command = ["python",os.path.basename(test_name_path)+".py"]
		(int_status,output,err) = run_process(command,False)

	elif (os.path.isfile(directory_test_path+"/default_tpips")):
		# Change flag for default_tpips
		os.chmod(directory_test_path+"/default_tpips", stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)
		command = "FILE="+test_file_path+" WSPACE="+os.path.basename(test_name_path)+" tpips "+directory_test_path+"/default_tpips"
		# Launch process
		(int_status,output,err) = run_process(command,True)

	elif (os.path.isfile(directory_test_path+"/default_test")):
		# Change flag for default_test
		os.chmod(directory_test_path+"/default_test", stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)
		# test_name=file
		# upper=FILE
		upper = os.path.basename(test_name_path).upper()
		command = "FILE="+test_file_path+" WSPACE="+os.path.basename(test_name_path)+" NAME="+upper+" "+directory_test_path+"/default_test 2>"+err_file_path
		(int_status,output,err) = run_process(command,True)

	elif (os.path.isfile(directory_test_path+"/default_pyps.py")):
		# Change flag for default_pyps.py
		os.chmod(directory_test_path+"/default_pyps.py", stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)
		command = "FILE="+test_file_path+" WSPACE="+os.path.basename(test_name_path)+" python "+directory_test_path+"/default_pyps.py 2>"+err_file_path
		(int_status,output,err) = run_process(command,True)

	# Orphan status (there is source file, .result directory and test reference file but there is no script to execute (.tpips, .tpips2, etc...))
	else:
		int_status = orphan_status
		output = ''
		err = None

	if err != None:
		err_file_path_h = open(err_file_path, 'w')
		err_file_path_h.write(err)
		err_file_path_h.close()

		# Apply .flt file for tpips2
		if os.path.isfile(test_name_path+".flt") & tpips2 == True:
			command_flt = ["cat " + err_file_path + " | "+ test_name_path+'.flt']
			process_flt =  subprocess.Popen(command_flt,stdout=subprocess.PIPE,shell=True)
			output_flt = process_flt.stdout.read()
			output = output + "### stderr\n"+ output_flt

	return (int_status,output)

### Check to see if there is a multi-script or multi-source status ###
def multi_source_script(directory_test,test_root):
	multi_status = ''
	nb_source = 0
	nb_script = 0

	# check if there is several sources for one script
	for ext in extension:
		if os.path.isfile(directory_test+'/'+test_root+ext):
			nb_source = nb_source + 1

	# check if there is sevral script for one source
	for exe in script:
		if os.path.isfile(directory_test+'/'+test_root+exe):
			nb_script = nb_script + 1

	if nb_script > 1:
		multi_status = 'multi-script'
	elif nb_source > 1:
		multi_status = 'multi-source'

	return multi_status

#### Function which run tests and save result on result_log ######
def test_par4all(directory_test_path,test_file_path,log_file):
	# .result directory of the test to compare results
	test_file_path = test_file_path.strip('\n')
	(test_name_path, ext) = os.path.splitext(test_file_path)
	test_result_path = test_name_path + '.result'

	#output ref of the test
	test_ref_path = test_result_path + '/test'

	status = 'succeeded'

	#check that .result and reference of the test are present. If not, status is "skipped" 
	if (os.path.isdir(test_result_path) != True or (os.path.isfile(test_ref_path) != True and os.path.isfile(test_ref_path+'.'+arch) != True)):
		status ='skipped'
	# Test is in development or it is known like a bug
	elif (os.path.isfile(test_name_path+".bug") or os.path.isfile(test_name_path+".later")):
		status ='bug-later'
	else:
		# output of the test and error of the tests
		output_file_path = test_result_path+'/'+os.path.basename(test_name_path)+'.out'
		err_file_path = test_name_path + '.err'

		# go to the directory of the test
		os.chdir(directory_test_path)

		# remove old .database
		for filename in glob.glob(directory_test_path+'/*.database') :
			shutil.rmtree(filename,ignore_errors=True)

		for filename in glob.glob(par4ll_validation_dir+'/*.database') :
			shutil.rmtree(filename,ignore_errors=True) 

		# test
		(int_status,output) = command_test(directory_test_path,test_name_path,err_file_path,test_file_path)

		# filter out absolute path anyway, there may be some because of
		# cpp or other stuff run by pips, even if relative path names are used.
		output = output.replace(directory_test_path,'.')

		if (os.path.isfile(err_file_path) == True):
			# copy error file on RESULT directories of par4all validation
			new_err = err_file_path.replace(par4ll_validation_dir,'')
			shutil.move(err_file_path,log_file_path+'/RESULT/'+new_err.replace('/','_'))

		if (int_status == timeout_value):
			status = 'timeout'
		elif (int_status == orphan_status):
			status = 'orphan'
		elif(int_status != 0):
			status = "failed"
		else:
			if(os.path.isfile(test_ref_path+'.'+arch) == True):
				ref_path = test_ref_path+'.'+arch
			else:
				ref_path = test_ref_path

			reference_filtered_path = ref_path

			out_filtered = output
			reference_filtered = open(ref_path).read()

			# let's apply some filter on output if define by user
			# First try to apply a test_and_out filter on both expected and obtained output
			# Else try to apply test and out filter on test and out output
			if (os.path.isfile(test_result_path + "/test_and_out.filter")):
				# apply the user define filter on both the reference
				os.chmod(test_result_path + "/test_and_out.filter", stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)

				output_file_filter_path = test_result_path+'/'+os.path.basename(test_name_path)+'.out.filtered'
				output_file_filter_h = open(output_file_filter_path,'w')
				output_file_filter_h.write ('%s' % (output))
				output_file_filter_h.close()

				sub_process_ref = subprocess.Popen([test_result_path + "/test_and_out.filter",reference_filtered_path], stdout=subprocess.PIPE)
				reference_filtered = sub_process_ref.communicate()[0]
				int_status = sub_process_ref.returncode

				sub_process_out = subprocess.Popen([test_result_path + "/test_and_out.filter",output_file_filter_path], stdout=subprocess.PIPE)
				out_filtered = sub_process_out.communicate()[0]
				int_status = sub_process_out.returncode

				# Write new "test reference" filtered
				reference_filter_path = ref_path + '.filtered'
				reference_filter_h = open(reference_filter_path,'w')
				reference_filter_h.write ('%s' % (reference_filtered))
				reference_filter_h.close()

				out_is_filtered=1
				ref_is_filtered=1

			else:
				# apply the user define filter on the reference
				if (os.path.isfile(test_result_path + "/test.filter")):
					os.chmod(test_result_path + "/test.filter", stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)

					sub_process_ref = subprocess.Popen([test_result_path + "/test_and_out.filter",reference_filtered_path], stdout=subprocess.PIPE)
					reference_filtered = sub_process_ref.communicate()[0]
					int_status = sub_process_ref.returncode

					# Write new "test reference" filtered
					reference_filter_path = ref_path + '.filtered'
					reference_filter_h = open(reference_filter_path,'w')
					reference_filter_h.write ('%s' % (reference_filtered))
					reference_filter_h.close()
					ref_is_filtered=1

				# apply the user define filter on the output
				if (os.path.isfile(test_result_path + "/out.filter")):
					os.chmod(test_result_path + "/out.filter", stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)

					sub_process_out = subprocess.Popen([test_result_path + "/test_and_out.filter",output_file_filter_path], stdout=subprocess.PIPE)
					out_filtered = sub_process_out.communicate()[0]
					int_status = sub_process_out.returncode

					out_is_filtered=1

			output_file_h = open(output_file_path,'w')
			output_file_h.write ('%s' % (out_filtered))
			output_file_h.close()

			# Diff between output filtered and reference filtered
			if (reference_filtered.replace(" ","").replace('\n','') != out_filtered.replace(" ","").replace('\n','')):
				status = 'changed'
			else:
				#status of the test
				status = 'succeeded'

	# Check to see if there is multi-script or multi-source
	multi_status = multi_source_script(directory_test_path,os.path.basename(test_name_path))

	if multi_status != '':
		status = status.replace(':','')+'_'+multi_status

	# Write status
	write_log(status,log_file,test_file_path)

	# Return to validation Par4All
	os.chdir(p4a_root)
	return status

###### Validate only test what we want ######
def valid_par4all():
	os.chdir(p4a_root)
	if os.path.isfile(log_file_path+'/par4all_validation.txt'):
		f = open(log_file_path+"/par4all_validation.txt")
	else:
		print ('No par4all_validation.txt file in %s. Create one before launch validation par4all'%(p4a_root))
		exit()

	# Create directory for result
	if (os.path.isdir(log_file_path+"/RESULT") == True):
		shutil.rmtree(log_file_path+"/RESULT", ignore_errors=True)
	os.mkdir(log_file_path+"/RESULT")

	if os.path.isfile(log_file_path+'/p4a_log.txt'):
		os.remove(log_file_path+"/p4a_log.txt")

	test_list = list()
	log_file = log_file_path+"/p4a_log.txt"

	# Open the file where par4all tests are:
	for line in f:
		# Check line is not empty
		if len(line.strip('\n').replace(' ','')) != 0 :
			# In case of the test is written like Directory_test\test.f instead os Directory_test/test.f
			line = line.replace('\\','/')
			# delete .f, .c and .tpips of the file name
			(root, ext) = os.path.splitext(line)

			# split to have: link to folder of the test and name of the test
			directory=root.split("/")
			directory_test = par4ll_validation_dir

			for j in range(0,len(directory)-1):
				directory_test = directory_test+'/'+directory[j]

			ext = ext.strip('\n').strip(' ')

			# File to test
			file_tested = (par4ll_validation_dir+line).strip('\n')

			# Check that test can be add to the test_list
			if(ext in extension):
				if (os.path.isdir(directory_test) != True ):
					print('%s is not a directory into packages/PIPS/validation'%(directory_test.replace(par4ll_validation_dir+'/','')))
				elif (os.path.isfile(par4ll_validation_dir+line.strip('\n').strip(' ')) != True):
					print('%s is not a file into packages/PIPS/validation/'%(line.strip('\n')))
				else:
					test_list.append(file_tested)
			elif (ext == '.result'):
					test_name_list = search_file(file_tested.replace('.result',''))
					for test_name_path in test_name_list:
						test_list.append(file_tested)
			else:
				print ("To test %s, use an extension like %s\n"%(line.strip('\n'),extension))

	f.close()

	# Launch test in multithread
	multithread_test(test_list,log_file)

	f_log=open(log_file)
	for line in f_log:
		print line.strip('\n')
	f_log.close()

	count_failed_warn(log_file)

#### Multiprocessing test ####
def multi_test(result_dir,directory_test,log_file):
	test_name_list = search_file(result_dir.replace('.result',''))
	for test_name_path in test_name_list:
		test_par4all(directory_test,test_name_path,log_file)

### .result test to test ####
def result_test(directory_test,resultdir_list,log_file):
	for resultdir in resultdir_list:
		multi_test(resultdir,directory_test,log_file)

#### Directory to test - Recursive to enter in subdirectories ####
def recursive_dir_test(dirlist,log_file,new_dir_list):
	# Check that list is not empty, so there is directory to test
	if (len(dirlist) != 0):
		i = 0
		# List of subdirectories to test
		dir_sublist = list()

		# Browse directory
		for i in range(0,len(dirlist)):
			resultdir_list = list()
			directory_test = dirlist[i]

			print (('# Considering %s')%(directory_test.replace(par4ll_validation_dir,'').strip('\n')))

			if os.path.isdir(directory_test):
				# List file/directories to test, subdirectories and skipped status (no correspondig .result folder)
				listing = os.listdir(directory_test)
				for dirfile in listing:
					dir_sublist,resultdir_list = subdir_list_test(directory_test,log_file,resultdir_list,dirfile,dir_sublist)

				# Test
				result_test(directory_test,resultdir_list,log_file)
			else:
				print ('None valid directory to test: %s does not exist'%(directory_test.replace(par4ll_validation_dir,'').strip('\n')))
				
			print (('# %s Finished')%(directory_test.replace(par4ll_validation_dir,'').strip('\n')))

		# Lock to have only one process who add dir to test
		lock = multiprocessing.Lock()
		lock.acquire()
		for dir in dir_sublist:
			new_dir_list.append(dir)
		lock.release()

### Return number of failed, warning and test ####
def count_failed_warn(log_file):
	nb_warning = 0
	nb_failed  = 0
	nb_test = 0

	if os.path.exists(log_file):
		log_file_h = open(log_file,'r')
		readline_log_file = log_file_h.readlines()

		# Number of test
		nb_test = len(readline_log_file)

		log_file_str = str(readline_log_file)

		# Number of warning
		i = 0
		for i in range(0,len(warning)):
			nb_warning = nb_warning + int(log_file_str.count(warning[i]+':'))
			nb_warning = nb_warning + int(log_file_str.count(warning[i]+'_multi-script:'))
			nb_warning = nb_warning + int(log_file_str.count(warning[i]+'_multi-source:'))

		# Number of failed
		i = 0
		for i in range(0,len(failed)):
			nb_failed = nb_failed + int(log_file_str.count(failed[i]+':'))
			nb_failed = nb_failed + int(log_file_str.count(failed[i]+'_multi-script:'))
			nb_failed = nb_failed + int(log_file_str.count(failed[i]+'_multi-source:'))

		log_file_h.close()

		if (log_file_path+'/diff.txt' != log_file):
			print('%s failed and %s warning %s in %s tests'%(nb_failed,nb_warning,warning,nb_test))
		else:
			print('%i tests are not done by --p4a options or their status is "skipped"'%(nb_test))

### Check subdirectories to test and skipped status (no correspondig .result folder) ####
def subdir_list_test(directory_test,log_file,resultdir_list,dirfile,dir_sublist):
	(root, ext) = os.path.splitext(dirfile)
	# Is it a directory?
	if os.path.isdir(directory_test+'/'+dirfile):
		if (ext == '.result'):
			resultdir_list.append(directory_test+'/'+dirfile)
		elif (ext == '.sub'):
			dir_sublist.append(directory_test+'/'+dirfile)
		else:
			# Is it in the makefile?
			for line in open(directory_test+"/Makefile"):
				if ((dirfile in line) and ("D.sub" in line)) :
					dir_sublist.append(directory_test+'/'+dirfile)
	# This is a file
	else:
		if ext in extension:
			test_result_path = directory_test+'/'+root + '.result'

			#output ref of the test
			test_ref_path = test_result_path + '/test'

			#check that .result and reference of the test are present. If not, status is "skipped" 
			if (os.path.isdir(test_result_path) != True or (os.path.isfile(test_ref_path) != True and os.path.isfile(test_ref_path+'.'+arch) != True)):
				write_log('skipped',log_file,directory_test+'/'+dirfile)

	return dir_sublist,resultdir_list

### List tests in directories and subdirectories and check that it's not present in par4all_validation.txt ####
def recursive_list_test(dir_list,par4all_string):
	# Check that list is not empty, so there is directory to test
	if (len(dir_list) != 0):
		i = 0

		# Browse directory
		for i in range(0,len(dir_list)):
			directory_test = dir_list[i]
			# List of subdirectories to test
			dir_sublist = list()

			# Check subdirectories to test and skipped status (no correspondig .result folder)
			resultdir_list = list()
			listing = os.listdir(directory_test)
			for dirfile in listing:
				dir_sublist,resultdir_list = subdir_list_test(directory_test,log_file_path+"/diff.txt",resultdir_list,dirfile,dir_sublist)

			for result_dir in resultdir_list:
				# Search a correspondig file of .result folder
				test_name_list = search_file(result_dir.replace('.result',''))
				for test_name_path in test_name_list:
					# Test is find. Check that it is present into par4all_validation.txt
					test_string = test_name_path.replace(par4ll_validation_dir,'').strip('\n')

					# Is test is in par4all_validation.txt ?
					if int(par4all_string.count(test_string)) == 0:
						diff_test_h = open(log_file_path+'/diff.txt','a')
						diff_test_h.write(test_name_path.replace(par4ll_validation_dir,'')+'\n')
						diff_test_h.close()

		recursive_list_test(dir_sublist,par4all_string)

### Find file (c, fortran, etc...) of the corresponding .result test ###
def search_file(filename):
	file_found = 0
	files = list()

	# List all tests file with corresponding .result folder
	for ext in extension:
		if os.path.exists(filename+ext):
			files.append(filename+ext)
			file_found = 1

	if file_found == 0:
		files.append(filename+'.result')

	return files

### Directories and subdirectories to test ###
def dir_subdir_test(dir_list,log_file):
	# list of launched process
	process_list = list()
	i = 0

	while i < len(dir_list):
		# Create temporary dir_list. This list will be tested after.
		dir_list_temp = list()
		for j in range(0,nb_cpu):
			if i < len(dir_list):
				dir_list_temp.append(dir_list[i])
				i = i+1

		# Create a manager to update dir_list
		manager = multiprocessing.Manager()
		new_dir_list = manager.list()

		# Multithread test
		for dir in dir_list_temp:
			process = multiprocessing.Process(target=recursive_dir_test, args=([dir],log_file,new_dir_list))
			process.start()
			process_list.append(process)

		# Wait for all threads to complete
		for thread in process_list:
			thread.join()

		# add subdirectories to the list of test
		dir_list += new_dir_list

### Multithread test for par4all and test options ###
def multithread_test(test_list,log_file):
	# list of launched process
	process_list = list()
	i = 0

	while i < len(test_list):
		# Create temporary test_list. This list will be tested after.
		test_list_temp = list()
		for j in range(0,nb_cpu):
			if i < len(test_list):
				test_list_temp.append(test_list[i])
				i = i+1

		# Multithread test
		for test in test_list_temp:
			test_array=test.split("/")

			# Directory of the test
			directory_test = ''
			for j in range(0,len(test_array)-1):
				directory_test = directory_test+'/'+test_array[j]

			print (('# Considering %s')%(test.replace(par4ll_validation_dir,'').strip('\n')))

			process = multiprocessing.Process(target=test_par4all,args=(directory_test,test,log_file))
			process.start()
			process_list.append(process)

		# Wait for all threads to complete
		for thread in process_list:
			thread.join()

###### Validate all tests (done by "default" file) ######
def valid_pips():
	global dir_list
	os.chdir(p4a_root)

	# Create directory for result
	if (os.path.isdir(log_file_path+"/RESULT") == True):
		shutil.rmtree(log_file_path+"/RESULT", ignore_errors=True)
	os.mkdir(log_file_path+"/RESULT")

	# Default file (where directories to test is listed
	default_file_path = par4ll_validation_dir+"defaults"

	default_file = open(default_file_path)

	if os.path.isfile(log_file_path+'/pips_log.txt'):
		os.remove(log_file_path+"/pips_log.txt")

	# List all directories that we must test
	for line in default_file:
		if (not re.match('#',line)):
			line  = line.strip('\n')
			dir_list.append(par4ll_validation_dir+line)

	# Multithreading test
	if len(dir_list) != 0:
		dir_subdir_test(dir_list,log_file_path+'/pips_log.txt')
	else:
		print "No folder to test according to"+par4ll_validation_dir+"defaults"

	count_failed_warn(log_file_path+'/pips_log.txt')
	default_file.close()

###### Diff between p4a and pips options ######
def diff():
	os.chdir(p4a_root)
	# Read default file to build a file with all tests
	default_file = open(par4ll_validation_dir+"defaults")

	diff_file = open(log_file_path+'/diff.txt','w')
	diff_file.close()

	if os.path.isfile(log_file_path+'/par4all_validation.txt'):
		par4all_h = open(log_file_path+"/par4all_validation.txt")
		par4all_string = par4all_h.read()
		par4all_string = par4all_string.replace('\\','/').strip('\n')
		par4all_h.close()
	else:
		par4all_string = ''

	# Parse all tests done by default file in pips validation and build a file with all tests which are not written in par4all_validation.txt
	for line in default_file:
		if (not re.match('#',line)):
			line  = line.strip('\n')
			if os.path.isdir(par4ll_validation_dir+line):
				dir_list = [par4ll_validation_dir+line]
				recursive_list_test(dir_list,par4all_string)
			else:
				print ('None valid directory to list: %s does not exist'%(line.strip('\n'))) 

	count_failed_warn(log_file_path+'/diff.txt')

###### Validate all tests of a specific directory ################
def valid_dir(arg_dir):
	global dir_list
	os.chdir(p4a_root)
	if os.path.isfile(log_file_path+'/directory_log.txt'):
		os.remove(log_file_path+"/directory_log.txt")

	# Create directory for result
	if (os.path.isdir(log_file_path+"/RESULT") == True):
		shutil.rmtree(log_file_path+"/RESULT", ignore_errors=True)
	os.mkdir(log_file_path+"/RESULT")

	# Build directory list to test
	for directory_name in arg_dir:
		# Is it a valid directory?
		if (os.path.isdir(par4ll_validation_dir+directory_name) != True):
			print ("%s does not exist or it's not a repository"%(directory_name))
		else:
			dir_list.append(par4ll_validation_dir+directory_name)

	# Multithreading test
	if len(dir_list) != 0:
		dir_subdir_test(dir_list,log_file_path+"/directory_log.txt")
	else:
		print ('None valid folder to test in %s'%(arg_dir))

	count_failed_warn(log_file_path+'/directory_log.txt')

###### Validate all desired tests ################
def valid_test(arg_test):
	os.chdir(p4a_root)
	# Create directory for result
	if (os.path.isdir(log_file_path+"/RESULT") == True):
		shutil.rmtree(log_file_path+"/RESULT", ignore_errors=True)
	os.mkdir(log_file_path+"/RESULT")

	if os.path.isfile(log_file_path+'/test_log.txt'):
		os.remove(log_file_path+"/test_log.txt")
	log_file = log_file_path+'/test_log.txt'

	test_list = list()

	#read the tests
	for i in range(0,len(arg_test)):
		test_array=arg_test[i].split("/")

		# Directory of the test
		directory_test = par4ll_validation_dir
		for j in range(0,len(test_array)-1):
			directory_test = directory_test+'/'+test_array[j]

		# File to test
		file_tested = directory_test+'/'+test_array[len(test_array)-1]
		(root, ext) = os.path.splitext(test_array[len(test_array)-1])

		# Check that directory and test exist
		if (os.path.isdir(directory_test) != True):
			print('%s is not a directory into packages/PIPS/validation'%(directory_test.replace(par4ll_validation_dir+'/','')))
		elif (os.path.isfile(file_tested) != True and ext != '.result'):
			print('%s is not a file into packages/PIPS/validation/%s'%(arg_test[i],directory_test.replace(par4ll_validation_dir,'')))
		else:
			# Check that extension of the file is OK
			if(ext in extension):
				test_list.append(file_tested)
			elif (ext == '.result'):
				test_name_list = search_file(file_tested.replace('.result',''))
				for test_name_path in test_name_list:
					test_list.append(file_tested)
			else:
				print('%s : Not done (extension must be %s or a valid .result folder)'%(arg_test[i],extension))

	# Launch test in multithread
	multithread_test(test_list,log_file)

	if os.path.isfile(log_file):
		f=open(log_file)
		for line in f:
			print line.strip('\n')
		f.close()
	else:
		print ("%s does not exist"%(log_file))

###### Launch make validate of pips validation ################
def make_validate(options):
	os.chdir(par4ll_validation_dir)

	# Make 
	command = ["make"]

	# Add options for make
	for opt in options:
		command.append("-"+opt)

	# Target of makefile
	command.append("validate-out")

	# Launch validation
	process_valid = subprocess.Popen(command, stdout=subprocess.PIPE)
	output = process_valid.communicate()[0]

	makeval_file = open(log_file_path+'/make_validate.txt','w')
	makeval_file.write(output)
	makeval_file.close()

	# String to print
	string_command = str(command)
	string_command = string_command.replace("[","").replace("]","").replace("'","").replace(",","")

	print ('Output of the "'+string_command+'" is in '+log_file_path+'/make_validate.txt')

### return test and directory for filter options ###
def test_dir(line):
	# Remove last element corresponding to run-time
	line = line.split(" ")
	if (len(line) > 2):
		line.pop()

	# Find test name and directory of the test
	line[len(line)-1] = line[len(line)-1].replace('\\','/')
	test_index =  line[len(line)-1].rfind('/')
	test = line[len(line)-1][test_index+1:].strip(' ').strip('\n')
	dir = line[len(line)-1][:test_index].strip(' ').strip('\n')
	status = line[0]

	return (status,dir,test)

#### Rewrite validation.out to have a good format for pips/par4all validation team ################
def filter_makeval():
	file = par4ll_validation_dir+"/validation.out"
	if (os.path.isfile(par4ll_validation_dir+"/validation.out")):
		# Open the file to browse
		validout_file = open(par4ll_validation_dir+"/validation.out")

		# The new file with desired content
		valid_file = open(log_file_path+'/pips_valid.txt',"w")

		multi_source_list = set() # multi-source list
		multi_script_list = set() # multi-script list

		# Browse file to find multi-script and multi-source
		for line in validout_file:
			if (len(line.strip(' ').strip('\n')) != 0):
				(status,dir,test) = test_dir(line)
				# Remove multi-script and multi-source because there are tested
				if 'multi-source' in line:
					multi_source_list.add(dir+'/'+test)
				elif 'multi-script' in line:
					multi_script_list.add(dir+'/'+test)

		validout_file.close
		validout_file = open(par4ll_validation_dir+"/validation.out")

		# Browse file
		for line in validout_file:
			if (not 'multi-source' in line) and (not 'multi-script' in line):
				if (len(line.strip(' ').strip('\n')) != 0):
					(status,dir,test) = test_dir(line)
					# Check that it's not a default_pyps/test/tpips test
					if (not "default_pyps" in line) and (not "default_test" in line) and (not "default_tpips" in line):
						# Replace bug or later in bug-later
						if re.match('bug:',status) :
							status = status.replace(status,"bug-later:")
						elif re.match('later:',status):
							status = status.replace(status,"bug-later:")

						# orphan status but know like a bug or in development
						elif re.match('orphan:',status) and (os.path.isfile(par4ll_validation_dir+'/'+dir+'/'+test+".bug") or os.path.isfile(par4ll_validation_dir+'/'+dir+'/'+test+".later")):
							status = status.replace(status,"bug-later:")
						elif re.match('passed:',status) :
							status = status.replace(status,"succeeded:")

						# If test has another status (multi-source or multi-script), add multi-X at this status
						if (dir+'/'+test) in multi_script_list:
							status = status.replace(':','')+'_multi-script:'
						elif (dir+'/'+test) in multi_source_list:
							status = status.replace(':','')+'_multi-source:'

						# Find the extension of the test name
						ext_found = 0
						for ext in extension:
							file_test = par4ll_validation_dir+'/'+dir+'/'+test+ext
							if os.path.isfile(file_test):
								test = test+ext
								ext_found  = 1
								# Write in new file
								valid_file.write(status + ' ' +dir+'/'+test+'\n')

						if ext_found == 0 and os.path.isdir(par4ll_validation_dir+'/'+dir+'/'+test+'.result'):
							test = test+'.result'
							# Write in new file
							valid_file.write(status + ' ' +dir+'/'+test+'\n')

		validout_file.close
		valid_file.close
	else:
		print ("No validation.out file in par4all/packages/PIPS/validation folder. Launch --makeval option to have it.")

###################### Main -- Options #################################
def main():
	usage = "usage: python %prog [options]"
	parser = optparse.OptionParser(usage=usage)
	parser.add_option("--pips", action="store_true", dest="pips", help = "Validate tests which are given by default file (in packages/PIPS/validation)")
	parser.add_option("--p4a", action="store_true", dest="par4all", help = "Validate tests which are given by par4all_validation.txt (which must be previously created in par4all/src/validation)")
	parser.add_option("--diff", action="store_true", dest="diff", help = "List tests that are done with pips option but not with p4a option")
	parser.add_option("--dir", action="store_true", dest="dir", help = "Validate tests which are located in packages/PIPS/validation/directory_name")
	parser.add_option("--test", action="store_true", dest="test", help = "Validate tests given in argument")
	parser.add_option("--makeval", action="store_true", dest="makeval", help = "Launch 'make [options] validate-out' of pips validation. Options are make options without '-'. Example of usage: ./p4a-validate_class.py --makeval j4 l")
	parser.add_option("--filter", action="store_true", dest="filter", help = "Rewrite validation.out to have the desired file of pips/par4all validation team. Can be used with --makeval option")
	(options, args) = parser.parse_args()

	#set all locale categories to C (English), to make the test results consistent to match
	#the references. This is needed because the test references have been defined using 
	#this environment variable, so some shell commands used in the tests
	#such as 'cat */*.c' will be done in the same order as the references
	os.putenv('LC_ALL', 'C')

	if options.pips:
		vc = valid_pips()
		print('Result of the tests are in pips_log.txt')

	elif options.par4all:
		vc = valid_par4all()
		print('Result of the tests are in p4a_log.txt')

	elif options.diff:
		vc = diff()
		print('Tests which are not done by --p4a options are into diff.txt file')

	elif options.dir:
		if (len(args) == 0):
			print("You must enter the name of the directories you want to test")
			exit()

		vc = valid_dir(args)
		print('Result of the tests are in directory_log.txt')

	elif options.test:
		if (len(args) == 0):
			print("You must enter the name of the tests you want to test")
			exit()

		vc = valid_test(args)

	elif options.makeval:
		vc = make_validate(args)
		print ('Summary of the validation is in par4all/packages/PIPS/validation/SUMMARY')

	else:
		if (not options.filter):
			# Help
			print subprocess.Popen(["python","p4a_validate_class.py","-h"], stdout=subprocess.PIPE).communicate()[0]

	if options.filter:
		vc = filter_makeval()
		print('Result of the filter are in pips_valid.txt')

	os.chdir(os.getcwd())

# If this programm is independent it is executed:
if __name__ == "__main__":
    main()

