#!/bin/sh

###
### Par4All Environment
### 
#
#  Run 'source <this file>' from your sh-compatible shell.
#
##########################################################

# Par4All source root. Might point to P4A_DIST if 
# sources are not installed.
export P4A_ROOT='$ROOT'

# Path to the Par4All installation.
export P4A_DIST='$DIST'

# Location of the Par4All_accelerator files.
export P4A_ACCEL_DIR=$$P4A_ROOT/src/p4a_accel

# Location of the Par4All configuration files.
export P4A_ETC=$$P4A_DIST/etc

# The Fortran compiler to use.
export PIPS_F77=gfortran

# Update PATH.
append_PATH () { if ! echo $$PATH | /bin/egrep -q "(^|:)$$1($$|:)"; then PATH=$$1:$$PATH; fi }
append_PATH $$P4A_DIST/bin
unset append_PATH

# Update libraries search paths.
append_PKG_CONFIG_PATH () { if ! echo $$PKG_CONFIG_PATH | /bin/egrep -q "(^|:)$$1($$|:)"; then PKG_CONFIG_PATH=$$1:$$PKG_CONFIG_PATH; fi }
append_PKG_CONFIG_PATH $$P4A_DIST/lib/pkgconfig
unset append_PKG_CONFIG_PATH

# Update Python module search path with PIPS Python bindings (PyPS).
append_PYTHONPATH () { if ! echo $$PYTHONPATH | /bin/egrep -q "(^|:)$$1($$|:)"; then PKG_CONFIG_PATH=$$1:$$PYTHONPATH; fi }
append_PYTHONPATH $$(echo $$P4A_DIST/lib/python*/site-packages/pips)
unset append_PYTHONPATH
