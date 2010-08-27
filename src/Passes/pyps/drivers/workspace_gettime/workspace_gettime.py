import pyps
import os
import sys

hfile= r"""#ifndef PYPS_GETTIME_H
#define PYPS_GETTIME_H

#define main _pyps_main

#endif //PYPS_GETTIME_H"""

cfile= r"""#undef main

#include <sys/time.h>
#include <stdio.h>

extern int _pyps_main(int argc, char **argv);

int main (int argc, char **argv)
{
	struct timeval time1, time2;
	int res;
	
	gettimeofday(&time1, NULL);

	_pyps_main(argc, argv);
	
	gettimeofday(&time2, NULL);
	
	long diff = (time2.tv_sec-time1.tv_sec)*1000000 + (time2.tv_usec-time1.tv_usec);
	
	FILE *out = fopen("_pyps_time.tmp", "w");
	if (out) {
		fprintf(out, "%ld\n", diff);
		fclose(out);
	}
	fprintf(stderr, "time for %s: %ld\n", argv[0], diff);

	return 0;
}
"""

""" When going to compile, edit all the c files to add the macros
		corresponding to fortran symbols"""
class workspace:
	def __init__(self, ws, source, *args, **kwargs):
		pass

	def pre_goingToRunWith(self, files, outdir):
		""" Creating the file containing the new main """
		intrpath = "_pyps_main_gettime.c"
		with open(intrpath, 'w') as f:
			f.write(cfile)
		files.append(intrpath)

		""" Creating the file containing the macro """
		intrpath = "_pyps_main_gettime.h"
		with open(intrpath, 'w') as f:
			f.write(hfile)

		"""Editing the source files to add the include in them"""
		abspath = os.path.abspath(intrpath)
		for file in files:
			with open(file, 'r') as f:
				read_data = f.read()
			#Don't put the include more than once
			if read_data.find('\n#include "{0}"\n'.format(abspath)) != -1:
				continue
			with open(file, 'w') as f:
				f.write('/* Header automatically inserted by PYPS*/\n#include "{0}"\n\n'.format(abspath))
				f.write(read_data)

	def post_compile(self, *args, **kwargs):
		try:
			os.unlink("_pyps_main_gettime.c")
			os.unlink("_pyps_main_gettime.h")
		except OSError:
			pass
