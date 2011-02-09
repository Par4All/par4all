###
### Par4All Environment
###
#
#  Run 'source <this file>' from your csh-compatible shell.
#
##########################################################

# Par4All source root. Might point to P4A_DIST if
# sources are not installed.
setenv P4A_ROOT '$root'

# Path to the Par4All installation.
setenv P4A_DIST '$dist'

# Location of the Par4All_accelerator files.
setenv P4A_ACCEL_DIR '$accel'

# Location of the Par4All configuration files.
setenv P4A_ETC $$P4A_DIST/etc

# The Fortran 77 compiler to use.
setenv PIPS_F77 $fortran

# Update PATH.
setenv PATH $$P4A_DIST/bin:$$PATH

# Update pkgconfig search path
if ( -d $$P4A_DIST/lib/pkgconfig ) then
    if ( $$?PKG_CONFIG_PATH ) then
	setenv PKG_CONFIG_PATH $$P4A_DIST/lib/pkgconfig:$${PKG_CONFIG_PATH}
    else
	setenv PKG_CONFIG_PATH $$P4A_DIST/lib/pkgconfig
    endif
endif
if ( -d $$P4A_DIST/lib64/pkgconfig ) then
    if ( $$?PKG_CONFIG_PATH ) then
	setenv PKG_CONFIG_PATH $$P4A_DIST/lib64/pkgconfig:$${PKG_CONFIG_PATH}
    else
	setenv PKG_CONFIG_PATH $$P4A_DIST/lib64/pkgconfig
    endif
endif

# Update library search path
if ( -d $$P4A_DIST/lib ) then
    if ( $$?LD_LIBRARY_PATH ) then
	setenv LD_LIBRARY_PATH $$P4A_DIST/lib:$${LD_LIBRARY_PATH}
    else
	setenv LD_LIBRARY_PATH $$P4A_DIST/lib
    endif
endif
if ( -d $$P4A_DIST/lib64 ) then
    if ( $$?LD_LIBRARY_PATH ) then
	setenv LD_LIBRARY_PATH $${P4A_DIST}/lib64:$${LD_LIBRARY_PATH}
    else
	setenv LD_LIBRARY_PATH $$P4A_DIST/lib64
    endif
endif

# Update Python module search path for pyps
set PYPS_PATH=`pkg-config pips --variable=pkgpythondir`
if ( -d $$PYPS_PATH ) then
    if ( $$?PYTHONPATH ) then
	setenv PYTHONPATH $${PYPS_PATH}:$${PYTHONPATH}
    else
	setenv PYTHONPATH $$PYPS_PATH
    endif
endif
# check also in lib64 if it exists (for fedora distro)
set PYPS_PATH=`echo $$PYPS_PATH | sed -e 's/lib/lib64/'`
if ( -d $$PYPS_PATH ) then
    if ( $$?PYTHONPATH ) then
	setenv PYTHONPATH $${PYPS_PATH}:$${PYTHONPATH}
    else
	setenv PYTHONPATH $$PYPS_PATH
    endif
endif

# Update the Python module search path so python can find ply
set PLY_PATH=/usr/lib/python2.6/site-packages
if ( -d $$PLY_PATH/ply ) then
    if ( $$?PYTHONPATH ) then
	setenv PYTHONPATH $${PLY_PATH}:$${PYTHONPATH}
    else
	setenv PYTHONPATH $$PLY_PATH
    endif
endif
set PLY_PATH=/usr/lib/python2.7/site-packages
if ( -d $$PLY_PATH/ply ) then
    if ( $$?PYTHONPATH ) then
	setenv PYTHONPATH $${PLY_PATH}:$${PYTHONPATH}
    else
	setenv PYTHONPATH $$PLY_PATH
    endif
endif
set PLY_PATH=/usr/lib/python3.1/site-packages
if ( -d $$PLY_PATH/ply ) then
    if ( $$?PYTHONPATH ) then
	setenv PYTHONPATH $${PLY_PATH}:$${PYTHONPATH}
    else
	setenv PYTHONPATH $$PLY_PATH
    endif
endif

# update the quincallerie
rehash
