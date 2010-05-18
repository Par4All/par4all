#! /usr/bin/env python3.1

# -*- coding: utf-8 -*-

"""
Validation utility for Par4All

Add object oriented organization above PIPS validation.

Introduce the concept of validation class.
"""
import os, commands, string, re, optparse

class ValidationClass:

###### Init of the validation ######
  def __init__(self):
		self.p4a_root = os.environ.get("P4A_ROOT")
		
		if (not self.p4a_root):
			print ('You need to define P4A_ROOT environment variable')
			exit()

		self.PWD = os.environ.get("PWD")
		self.par4ll_validation_dir = self.p4a_root+'/packages/PIPS/validation/'

		# get default architecture and tpips/pips
		self.arch=commands.getoutput(self.p4a_root+"/run/makes/arch.sh")

#### Function which run tests and save result on result_log ######
  def test_par4all(self,directory_test_path,test_file_path,log_file,extension_file):
		# .result directory of the test to compare results
		test_file_path = test_file_path.strip('\n')
		(test_name_path, ext) = os.path.splitext(test_file_path)
		test_result_path = test_name_path + '.result'
	
		#output ref of the test
		test_ref_path = test_result_path + '/test'

		# log/summary of the results
		self.file_result = open(log_file,'a')
		status = 'succeeded'

		#check that .result and reference of the test are present. If not, status is "skipped" 
		if (os.path.isdir(test_result_path) != True or (os.path.isfile(test_ref_path) != True and os.path.isfile(test_ref_path+'.'+self.arch) != True)):
			status ='skipped'
		else:
			# output of the test and error of the tests
			output_file_path = test_result_path+'/'+os.path.basename(test_name_path)+'.out'
			err_file_path = test_name_path + '.err'
		
			# go to the directory of the test
	 		os.chdir(directory_test_path)

			commands.getstatusoutput("rm -rf *.database")

			if (os.path.isfile(test_name_path+".test")):
				(int_status, output) = commands.getstatusoutput(test_name_path+".test 2> "+err_file_path)

			elif (os.path.isfile(test_name_path+".py")):
				(int_status, output) = commands.getstatusoutput("python "+test_name_path+".py 2> "+err_file_path)

			elif (os.path.isfile(test_name_path+".tpips")):
				(int_status, output) = commands.getstatusoutput("tpips "+test_name_path+".tpips 2> "+err_file_path)

			elif (os.path.isfile(test_name_path+".tpips2")):
				(int_status, output) = commands.getstatusoutput("tpips "+test_name_path+".tpips2 2>&1")
		
			elif (os.path.isfile(directory_test_path+"/default_test")):
				# test_name=file
				# upper=FILE
				upper = os.path.basename(test_name_path).upper()
				(int_status, output) = commands.getstatusoutput("FILE="+test_file_path+" WSPACE="+os.path.basename(test_name_path)+" NAME="+upper+" "+directory_test_path+"/default_test 2>"+err_file_path)

			elif (os.path.isfile(directory_test_path+"/default_tpips")):
				(int_status, output) = commands.getstatusoutput("FILE="+test_file_path+" WSPACE="+os.path.basename(test_name_path)+" tpips "+directory_test_path+"/default_tpips 2>"+err_file_path)
		
			else:
				# Create a err file
				err_file_h = open(err_file_path,'w')
				
				commands.getstatusoutput("Delete "+os.path.basename(test_name_path)+" 2> /dev/null 1>&2")

				(int_status, output) = commands.getstatusoutput("Init -f "+test_file_path+" -d "+os.path.basename(test_name_path)+" 2> "+err_file_path)

				(int_status, output_display) = commands.getstatusoutput("while read module ; do Display -m  $module -w "+os.path.basename(test_name_path)+" ; done < "+os.path.basename(test_name_path)+".database/modules")
				err_file_h.write ('%s' % (output_display))

				commands.getstatusoutput("Delete "+os.path.basename(test_name_path)+" 2> /dev/null 1>&2")

				err_file_h.close()

			# filter out absolute path anyway, there may be some because of
  		# cpp or other stuff run by pips, even if relative path names are used.
			output = output.replace(test_file_path,'./'+os.path.basename(test_file_path))

			if (os.path.isfile(err_file_path) == True):
				# copy error file on RESULT directories of par4all validation
				commands.getstatusoutput("mv -f "+err_file_path+' $P4A_ROOT/src/validation/RESULT')
				os.rename(self.p4a_root+"/src/validation/RESULT/"+os.path.basename(err_file_path),self.p4a_root+"/src/validation/RESULT/"+os.path.basename(directory_test_path)+"_"+os.path.basename(err_file_path))
		
			if(int_status != 0):
				status = "failed"
			else:
				if(os.path.isfile(test_ref_path+'.'+self.arch) == True):
					ref_path = test_ref_path+'.'+self.arch
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
					commands.getstatusoutput('chmod +x ' + test_result_path + "/test_and_out.filter")

					output_file_filter_path = test_result_path+'/'+os.path.basename(test_name_path)+'.out.filtered'
					output_file_filter_h = open(output_file_filter_path,'w')
					output_file_filter_h.write ('%s' % (output))	
					output_file_filter_h.close()

					(int_status, reference_filtered) = commands.getstatusoutput(test_result_path + "/test_and_out.filter "+reference_filtered_path)
					(int_status, out_filtered) = commands.getstatusoutput(test_result_path + "/test_and_out.filter "+output_file_filter_path)

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
						commands.getstatusoutput("chmod +x "+ test_result_path + "/test.filter")
						(int_status, reference_filtered) = commands.getstatusoutput(test_result_path + "/test_and_out.filter "+reference_filtered_path)

						# Write new "test reference" filtered
						reference_filter_path = ref_path + '.filtered'
						reference_filter_h = open(reference_filter_path,'w')
						reference_filter_h.write ('%s' % (reference_filtered))
						reference_filter_h.close()
						ref_is_filtered=1

					# apply the user define filter on the output
					if (os.path.isfile(test_result_path + "/out.filter")):
						commands.getstatusoutput("chmod +x "+ test_result_path + "/out.filter")
						(int_status, out_filtered) = commands.getstatusoutput(test_result_path + "/test_and_out.filter "+output_file_filter_path)
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

		self.file_result.write ('%s: %s/%s%s\n' % (status,os.path.basename(directory_test_path),os.path.basename(test_name_path),extension_file))
		self.file_result.close()
	
		# Return to validation Par4All
		os.chdir(self.PWD)
		return status

###### Validate only test what we want ######
  def valid_par4all(self):
	
		if os.path.isfile('par4all_validation.txt'):
			f = open("par4all_validation.txt")
		else:
			print ('No par4all_validation.txt file in P4A_ROOT/src/validation. Create one before launch validation par4all')
			exit()

		# Create directory for result
		if (os.path.isdir("RESULT") == True):
			commands.getstatusoutput("rm -rf RESULT")
		os.mkdir("RESULT")

		if os.path.isfile('p4a_log.txt'):
			commands.getstatusoutput('rm -rf p4a_log.txt')

		nb_test = 0
		nb_failed = 0

    # Open the file where par4all tests are:		
		for line in f:
			# delete .f, .c and .tpips of the file name
			(root, ext) = os.path.splitext(line)

			# In case of the test is written like Directory_test\test.f instead os Directory_test/test.f
			if (root.find('\\') != -1):
				root = root.replace('\\','/')
				line = line.replace('\\','/')

			# split to have: link to folder of the test and name of the test
			directory=root.split("/")
			directory_test = self.par4ll_validation_dir + directory[0]
			
			print (('# Considering %s')%(line.strip('\n')))

			ext = ext.strip('\n')
			
			if(ext == '.c' or ext == '.F' or ext == '.f' or ext == '.f90'):
				if os.path.isdir(directory_test):
					# Run test
					nb_test = nb_test+1
					status = self.test_par4all(directory_test,self.par4ll_validation_dir+line,'p4a_log.txt',ext)
					if (status != "succeeded"):
						nb_failed = nb_failed+1
				else:
					print ('%s not accessible' % (directory_test))
			else:
				print ("To test %s, use an extension like .c, .f90, .f, .F\n"%(os.path.basename(self.par4ll_validation_dir+line).strip('\n')))

		f.close()
		print('%s failed in %s tests'%(nb_failed,nb_test))

###### Validate all tests (done by "default" file) ######
  def valid_pips(self):
		# Create directory for result
		if (os.path.isdir("RESULT") == True):
			commands.getstatusoutput("rm -rf RESULT")
		os.mkdir("RESULT")

		default_file_path = self.par4ll_validation_dir+"defaults"

		default_file = open(default_file_path)

		if os.path.isfile('pips_log.txt'):
			commands.getstatusoutput('rm -rf pips_log.txt')

		nb_test = 0
		nb_failed = 0

		for line in default_file:
				if (not re.match('#',line)):
					line  = line.strip('\n')
					directory_test = self.par4ll_validation_dir + line
					print (('# Considering %s')%(os.path.basename(directory_test)))

					for file_test in os.listdir(directory_test):
						(root, ext) = os.path.splitext(file_test)
						if(ext == '.c' or ext == '.F' or ext == '.f' or ext == '.f90'):
							nb_test = nb_test+1
							file_tested = directory_test + '/' + file_test
							status = self.test_par4all(directory_test, file_tested,'pips_log.txt',ext)
							if (status != "succeeded"):
								nb_failed = nb_failed+1

		print('%s failed in %s tests'%(nb_failed,nb_test))
		default_file.close()

###### Diff between p4a and pips options ######
  def diff(self):
		# Read default file to build a file with all tests
		default_file = open(self.par4ll_validation_dir+"defaults")

		default_test_file = open('diff.txt','w')
		default_test_file.close()
	
		nb_test = 0

		# Parse all tests done by pips and build a file with all tests
		for line in default_file:
				if (not re.match('#',line)):
					line  = line.strip('\n')
					directory_test = self.par4ll_validation_dir + line

					# Find tests
					for file_test in os.listdir(directory_test):
						(root, ext) = os.path.splitext(file_test)

						if(ext == '.c' or ext == '.F' or ext == '.f' or ext == '.f90'):
							# default_test depends of par4all_validation.txt
							default_test = line+'/'+os.path.basename(file_test)
							default_test_bis = line+'\ '+os.path.basename(file_test)
							find = 'no'
				
							nb_test = nb_test+1
							
							# Test is find. Check that it is present into par4all_validation.txt
							if os.path.isfile('par4all_validation.txt'):
								par4all = open("par4all_validation.txt")
								for line_p4a in par4all:
									if (default_test == line_p4a.strip('\n')):
										# Test is found
										find = 'yes'
										nb_test = nb_test - 1
									elif (default_test_bis.replace(" ","")== line_p4a.strip('\n')):
										# Test is found
										find = 'yes'
										nb_test = nb_test - 1
										default_test = default_test_bis.replace(" ","")

								par4all.close()

								if (find != 'yes'):
									default_test_h = open('diff.txt','a')
									default_test_h.write(default_test+'\n')
									default_test_h.close()
						
							# None par4all_validation.txt file
							else:
								default_test_h = open('diff.txt','a')
								default_test_h.write(default_test+'\n')
								default_test_h.close()

		print('%s tests are not done by --p4a options'%(nb_test))

###################### Main -- Options #################################
def main():
	usage = "usage: python %prog [options]"
	parser = optparse.OptionParser(usage=usage)
	parser.add_option("--pips", action="store_true", dest="pips", help = "Validate tests which are done by default file (in packages/PIPS/validation)")
	parser.add_option("--p4a", action="store_true", dest="par4all", help = "Validate tests which are done by par4all_validation.txt (which must be previously created in src/validation)")
	parser.add_option("--diff", action="store_true", dest="diff", help = "Show test that it's not done by p4a options")
	(options, args) = parser.parse_args()

	if options.pips:
		vc = ValidationClass().valid_pips()
		print('Result of the tests are in pips_log.txt')

	elif options.par4all:
		vc = ValidationClass().valid_par4all()
		print('Result of the tests are in p4a_log.txt')

	elif options.diff:
		vc = ValidationClass().diff()
		print('Tests which are not done by --p4a options are into diff.txt file')
	
	else:
		output = commands.getoutput("python p4a_validate_class.py -h")
		print(output)

# If this programm is independent it is executed:
if __name__ == "__main__":
    main()

