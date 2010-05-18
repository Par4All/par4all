#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Par4All Processing Class (convenience wrapper around pyps.workspace)
'''

import sys, os, re
from p4a_util import *
import pyps
	
class p4a_processor():

	files = []
	workspace = None
	main_filter = None

	def __init__(self, workspace = None, project_name = "", cppflags = "", verbose = False,
		files = [], filter_include = None, filter_exclude = None, accel = False):

		if workspace:
			self.workspace = workspace
		else:
			# This is because pyps.workspace.__init__ will test for empty strings...
			if project_name == None:
				project_name = ""
			if cppflags == None:
				cppflags = ""

			for file in files:
				if not os.path.exists(file):
					raise p4a_error("file does not exist: " + file)

			self.files = files
			
			# Create the PyPS workspace.
			self.workspace = pyps.workspace(files, name = project_name, activates = [], verboseon = verbose, cppflags = cppflags)
			self.workspace.set_property(FOR_TO_DO_LOOP_IN_CONTROLIZER = True,
				PRETTYPRINT_SEQUENTIAL_STYLE = "do")

		for module in self.workspace:
			module.prepend_comment(PREPEND_COMMENT = "/*\n * module " + module.name + "\n */\n")
				#+ " read on " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

		if accel:
			files.append(os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_stubs.c"))
		
		# Skip module name of P4A runtime.
		# Also filter out modules based on --include-modules and --exclude-modules.
		skip_p4a_runtime_and_compilation_unit_re = re.compile("P4A_.*|.*!")
		filter_include_re = None
		if filter_include:
			filter_include_re = re.compile(filter_include)
		filter_exclude_re = None
		if filter_exclude:
			filter_exclude_re = re.compile(filter_exclude)
		self.main_filter = (lambda module: not skip_p4a_runtime_and_compilation_unit_re.match(module.name)
			and (filter_exclude_re == None or not filter_exclude_re.match(module.name))
			and (filter_include_re == None or filter_include_re.match(module.name)))
		all_modules = self.workspace.filter(self.main_filter)

		all_modules.loop_normalize(
			# Loop normalize for the C language and GPU friendly
			LOOP_NORMALIZE_ONE_INCREMENT = True,
			LOOP_NORMALIZE_LOWER_BOUND = 0,
			# It is legal in the following by construction:
			LOOP_NORMALIZE_SKIP_INDEX_SIDE_EFFECT = True)

		all_modules.privatize_module()

	def filter_modules(self, filter_include = None, filter_exclude = None):
		filter_include_re = None
		if filter_include:
			filter_include_re = re.compile(filter_include)
		filter_exclude_re = None
		if filter_exclude:
			filter_exclude_re = re.compile(filter_exclude)
		filter = (lambda module: self.main_filter(module)
			and (filter_exclude_re == None or not filter_exclude_re.match(module.name))
			and (filter_include_re == None or filter_include_re.match(module.name)))
		return self.workspace.filter(filter)

	def parallelize(self, fine = False, filter_include = None, filter_exclude = None):
		all_modules = self.filter_modules(filter_include, filter_exclude)
		if fine:
			all_modules.internalize_parallel_code()
		else:
			all_modules.coarse_grain_parallelization()

	def gpuify(self, filter_include = None, filter_exclude = None):
		all_modules = self.filter_modules(filter_include, filter_exclude)

		all_modules.gpu_ify()

		# Isolate kernels by using the fact that all the generated kernels have
		# their name beginning with "p4a_":
		kernel_launcher_filter_re = re.compile("p4a_kernel_launcher_.*[^!]$")
		kernel_launchers = workspace.filter(lambda m: kernel_launcher_filter_re.match(m.name))

		# Add communication around all the call site of the kernels:
		kernel_launchers.kernel_load_store()
		kernel_launchers.gpu_loop_nest_annotate()

		# Inline back the kernel into the wrapper, since CUDA can only deal with
		# local functions if they are in the same file as the caller (by inlining
		# them, by the way... :-) )
		kernel_filter_re = re.compile("p4a_kernel_\\d+$")
		kernels = workspace.filter(lambda m: kernel_filter_re.match(m.name))
		kernels.inlining()

		# Display the wrappers to see the work done:
		kernel_wrapper_filter_re = re.compile("p4a_kernel_wrapper_\\d+$")
		kernel_wrappers = workspace.filter(lambda m: kernel_wrapper_filter_re.match(m.name))

	def ompify(self, filter_include = None, filter_exclude = None):
		self.filter_modules(filter_include, filter_exclude).ompify_code()

	def save(self, in_dir = None, prefix = "p4a_"):
		output_files = []
		self.workspace.save(in_dir, prefix)
		for file in self.files:
			(dir, name) = os.path.split(file)
			if in_dir:
				dir = in_dir
			output_file = os.path.join(dir, prefix + name)
			output_files += [ output_file ]
		return output_files

if __name__ == "__main__":
	print(__doc__)
	print("This module is not directly executable")

# What? People still use emacs? :-)
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
