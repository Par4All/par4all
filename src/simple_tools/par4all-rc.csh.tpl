###
### Par4All Environment
###
#
#  Run 'source <this file>' from your csh-compatible shell.
#
##########################################################

# If P4A_DIST is not externally defined, use a compile-time default value:
if (! $$?P4A_DIST ) then
    # Path to the Par4All installation.
    setenv P4A_DIST '$dist'
endif

# Par4All source root. Point to P4A_DIST if not defined:
if (! $$?P4A_ROOT ) then
  setenv P4A_ROOT $$P4A_DIST
endif

# Location of the Par4All_accelerator files.
setenv P4A_ACCEL_DIR $$P4A_DIST/$accel

# Location of the Par4All_scmp files.
setenv P4A_SCMP_DIR $$P4A_DIST/$scmp

# Location of the Par4All_astrad files.
setenv P4A_SCMP_DIR $$P4A_DIST/$astrad

# Location of the Par4All configuration files.
setenv P4A_ETC $$P4A_DIST/etc

# The Fortran 77 compiler to use.
setenv PIPS_F77 $fortran

# Location of PIPS, needed in case of relocation
setenv PIPS_ROOT $$P4A_DIST

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
# To make par4all relocatable, PYPS_PATH has to be updated to the new installation path
set PYPS_PATH_PREFIX=`echo $$PYPS_PATH |sed -e 's,\(.*\)\/lib\/python.*,\1,'`
if ( "$$PYPS_PATH_PREFIX" != "$$P4A_DIST" ) then
	set PYPS_PATH=`echo $$PYPS_PATH |sed -e 's,.*\(lib\/python.*\),'$$P4A_DIST'/\1,'`
endif
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

# update the quincallerie
rehash
