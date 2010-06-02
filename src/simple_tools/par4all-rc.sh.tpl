###
### Par4All Environment
### 
#
#  Run 'source <this file>' from your sh-compatible shell.
#
##########################################################

# Par4All source root. Might point to P4A_DIST if 
# sources are not installed.
export P4A_ROOT='$root'

# Path to the Par4All installation.
export P4A_DIST='$dist'

# Location of the Par4All_accelerator files.
export P4A_ACCEL_DIR='$accel'

# Location of the Par4All configuration files.
export P4A_ETC=$$P4A_DIST/etc

# The Fortran 77 compiler to use.
export PIPS_F77=$fortran

prepend_to_path_var () { perl -e "exit unless '$$2'; @p = grep { \$$_ and \$$_ ne '$$2' } split ':', \$$ENV{'$$1'}; print join ':', ('$$2', @p);"; }

# Update PATH.
export PATH=$$(prepend_to_path_var PATH $$P4A_DIST/bin)

# Update libraries search paths.
export PKG_CONFIG_PATH=$$(prepend_to_path_var PKG_CONFIG_PATH $$P4A_DIST/lib/pkgconfig)
# If libs path not in ld.so.conf.d, then ldconfig -p should not have it and
# we must add it to LD_LIBRARY_PATH.
if [ `ldconfig -p | grep $$P4A_DIST/lib | wc -l` = 0 ]; then
	export LD_LIBRARY_PATH=$$(prepend_to_path_var LD_LIBRARY_PATH $$P4A_DIST/lib)
fi

# Update the Python module search path so that python 3.1 locates python-ply.
PYTHONPATH=$$(prepend_to_path_var PYTHONPATH /usr/share/pyshared)
export PYTHONPATH

# Update the Python module search path for pyps.
NEW_PYTHON_PATH=$$(ls -d $$P4A_DIST/lib/python*/*-packages/pips 2>/dev/null | tail -1)
PYTHONPATH=$$(prepend_to_path_var PYTHONPATH $$NEW_PYTHON_PATH)
export PYTHONPATH

unset prepend_to_path_var
