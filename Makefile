################################################
## PolyLib Makefile.  Version : 5.0           ##
## Vincent Loechner, 1998/99/00.              ##
################################################
## Main targets of this makefile are :
## all, libs, install-libs, exec, install[-exec], uninstall, 
## 32ln-exec, 64ln-exec (to link executables without the extra suffix)
## Other targets : 32, 64, typecheck.

include vars.mk

############################################################
### IMPORTANT NOTE : The defines below should no longer be used:
### They are done by the configure script generating the file vars.mk

############################################################
### Which variants to build : 32 bits, 64 bits, or both.
### recommanded: both 32 and 64 libs, 64 executables.
##LIBS_TO_BUILD=32
##LIBS_TO_BUILD=64
#LIBS_TO_BUILD=32 64
#EXEC_TO_BUILD=32
##EXEC_TO_BUILD=64
##EXEC_TO_BUILD=32 64
#
### Define one of these to add an extra suffix to the executables
### This is usefull if you have multiple executables installed
### (32/64 bits or with/without GNU-MP for example)
##EXEC_EXTRA_SUFFIX = $(BITS)
##EXEC_EXTRA_SUFFIX = $(BITS).GMP
#
############################################################
### Type of integer to use (see ./ArithLib/ for details)
### ------------------64 bits integers----------------------
#LONG_BITS= 64
#
### 1. On most systems, 'long long int' is the 64 bits integer definition.
### If this is your case define the following :
#LONG_AFLAGS = -DLINEAR_VALUE_IS_LONGLONG -DLINEAR_VALUE_PROTECT_MULTIPLY \
# -DLINEAR_VALUE_ASSUME_SOFTWARE_IDIV
#
### 2. On some systems/processors (Alpha 21164 for example) 'long int'
### is the type for 64 bits integers
### If this is your case, define the following :
##LONG_AFLAGS = -DLINEAR_VALUE_IS_LONG -DLINEAR_VALUE_PROTECT_MULTIPLY \
## -DLINEAR_VALUE_ASSUME_SOFTWARE_IDIV
#
### ------------------32 bits integers----------------------
#INT_BITS= 32
#INT_AFLAGS = -DLINEAR_VALUE_IS_INT
#
### --------------------------------------------------------
### For the typechecking tests, this typedefs Value as 'char *' :
### (the executables won't run, this is just a compilation test)
#CHECK_BITS= chars
#CHECK_AFLAGS = -DLINEAR_VALUE_IS_CHARS -DLINEAR_VALUE_PROTECT_MULTIPLY
#
#
############################################################
### GNU-MP stuff
#
### Define these 4 lines if you want to use the GNU-mp (multiple precision)
### library when there is an overflow in Ehrhart
#GFLAGS = -DGNUMP
#EXTRA_LIBS = 
# 
### Define these 5 lines if your GNU-MP lib is not in the standard directory
##GMPDIR=/usr/local
##GMPLIB=$(GMPDIR)/lib
##GMPINCLUDE=$(GMPDIR)/include
##EXTRA_INCLUDES = -I$(GMPINCLUDE)
##EXTRA_LIBS += -L$(GMPLIB)
# 
############################################################
### Compiler stuff
#CC = /soft/purify/purify -best-effort -cache-dir=/tmp/purify gcc
#CC = gcc
#CFLAGS = -O4 -Wall -g 
##CFLAGS = -O4
#LDFLAGS = 
#EXTRA_LIBS += -lm
#
### In modern systems, just define ranlib as 'echo'
#RANLIB = echo
#
### LD flag to generate a shared library.
### if you use GNU ld define:
#SHAREDLIB_FLAG = -shared
### else define:
##SHAREDLIB_FLAG = -G
#
### set this if it is not defined by default (on HPUX for example)
##OSTYPE = HPUX
#
############################################################
### Installation directories, name of executables.
#
### make install puts everything here:
### (if you choose method 2. 3. or 4. below)
#INSTALLDIR = /usr/local
#BINDIR = $(INSTALLDIR)/bin
#LIBDIR = $(INSTALLDIR)/lib
#INCLUDEDIR = $(INSTALLDIR)/include
#MANDIR = $(INSTALLDIR)/man
#DOCSDIR = $(INSTALLDIR)/doc/packages/polylib-$(VERSION)
#
############################################################
### Installing the library
#
### Choose one of these 4 installation methods:
### 1. don't install the lib (just test the polylib)
### 2. install the shared lib
### 3. install the static lib
### 4. install both static and shared libs (recommended)
#
### 1. don't install polylib anywhere (static linking).
### define these 3 lines if you just want to test the library.
### If you choose this, you won't be able to build other packages
### such as the VisuDomain tool (well indeed you can by making
### naughty changes in its makefile :-)
##INSTALL_LIB = 
##EXEC_EXTRA_LIBS = $(LIB)

### Define this in combination with 2, 3, or 4 below
### (2. 3. and 4. -needed) general static/shared defines
### define this to link polylib with the executables:
##EXEC_EXTRA_LIBS += -lpolylib$(BITS)
### -optional: if the library is not installed in a standard path, define:
##EXEC_EXTRA_LIBS += -L$(LIBDIR)
#
### 2. (shared)
### Install the shared library:
##INSTALL_LIB = install-shared
### this is in most cases not necessary since it is the default:
##EXEC_EXTRA_LIBS += -Bdynamic
### On linux define this to run ldconfig after building the shared library:
##LDCONFIG = ldconfig
#
### 3. (static)
### Install the static library:
##INSTALL_LIB = install-static
#
### 4. (static AND shared)
### Install both static and shared libs:
##INSTALL_LIB = install-static install-shared
### this is in most cases not necessary since it is the default:
##EXEC_EXTRA_LIBS += -Bdynamic
### On linux define this to run ldconfig after building the shared library:
##LDCONFIG = ldconfig
#
#
### (2. and 4. -optional)
### Using shared libraries in non-standard paths.
### It is strongly discouraged to use this directory hard-coding
### in the executables. Anyway, it's not very usefull to have a
### shared object in a private directory. This can be used for
### testing purpose.
### On SVR4 (solaris, ...) to avoid defining LD_LIBRARY_PATH define:
##EXEC_LDFLAGS += -R$(LIBDIR)
### with GNU ld define:
##EXEC_LDFLAGS += -r$(LIBDIR)

mkinstalldirs = $(SHELL) ./mkinstalldirs

#############################################################
##-------------------END OF USER DEFINES-------------------##
#############################################################
##  you shouldn't need to change anything below this line  ##
#############################################################

#############################################################
## more defines
## where to put intermediate objects and executables:
OBJ_DIR = Obj.$(BITS).$(BUILD)-$(HOST)-$(OSTYPE)
LIB = $(OBJ_DIR)/$(PSTATIC)
#EXEC_EXTRA_LIBS = -L./$(OBJ_DIR) $(EXEC_EXTRA_LIBS)

POLYLIB_INC = ./include/polylib
POLYLIB_SRC = ./source
EXTRA_INCLUDES += -I ./include
ARITH_DIR = ./ArithLib
CFLAGS += $(EXTRA_INCLUDES) $(AFLAGS) $(EXTRA_FLAGS) 

PSTATIC = libpolylib$(BITS).a.$(VERSION)
PSHARED =  libpolylib$(BITS).$(SHEXT).$(VERSION)

PEXEC = \
	testlib \
	polytest \
	c2p \
	r2p \
	findv \
	pp \
	union_disjointe \
	union_convex \
	ehrhart \
	verif_ehrhart\
	Zpolytest\
	example

POLY_EXEC= $(PEXEC:%=$(OBJ_DIR)/%$(EXEC_EXTRA_SUFFIX))

CFILES= \
	errormsg.c \
	vector.c \
	matrix.c \
	polyhedron.c \
	polyparam.c \
	param.c \
	alpha.c \
	ehrhart.c \
	eval_ehrhart.c \
	SolveDio.c \
	Lattice.c \
	Matop.c \
	NormalForms.c \
	Zpolyhedron.c		
LIB_CFILES = $(CFILES:%=$(POLYLIB_SRC)/%)

PHEADERS= $(CFILES:%.c=$(POLYLIB_INC)/%.h) \
	$(POLYLIB_INC)/polylib.h $(POLYLIB_INC)/types.h \
	$(POLYLIB_INC)/arithmetique.h $(POLYLIB_INC)/arithmetic_errors.h \
	vars.mk

LIB_OBJECTS= $(CFILES:%.c=$(OBJ_DIR)/%.o) 


##############################################################
## default : make $LIBS_TO_BUILD libs and $EXEC_TO_BUILD exec
##############################################################
# main target
all::
	@echo "---------------------------------------------------"
	@echo "You can choose either:"
	@echo "'make libs' to build the libraries ($(LIBS_TO_BUILD))."
	@echo "'make install-libs' to install them (if necessary)."
	@echo "'make [all]' to build the libs, install them, and build the executables (this is the default)."
	@echo "'make install' to build and install everything."
	@echo "---------------------------------------------------"
all:: $(LIBS_TO_BUILD:%=%libs)
all:: $(LIBS_TO_BUILD:%=%install-libs)
all:: $(EXEC_TO_BUILD:%=%exec)
	@echo "---------------------------------------------------"
	@echo "Type 'make install' to install everything"
	@echo "---------------------------------------------------"

$(INT_BITS):
	$(MAKE) "BITS=$(INT_BITS)" "AFLAGS=$(INT_AFLAGS)" default

$(LONG_BITS):
	$(MAKE) "BITS=$(LONG_BITS)" "AFLAGS=$(LONG_AFLAGS)" default

$(GMP_BITS):
	$(MAKE) "BITS=$(GMP_BITS)" "AFLAGS=$(GMP_AFLAGS)" default

typecheck:
	$(MAKE) "BITS=$(CHECK_BITS)" "AFLAGS=$(CHECK_AFLAGS)" default

###########################################################
## called from all : make lib, install it, and build executables
###########################################################
default:: libs
	@echo "---------------------------------------------------"
	@echo "Successfully built Library."
	@echo "The lib now needs to be installed in order to build executables"
	@echo "---------------------------------------------------"
default:: $(INSTALL_LIB)
	@echo "---------------------------------------------------"
	@echo "Successfully installed Library."
	@echo "---------------------------------------------------"
default:: allexec

###########################################################
## Install/UnInstall
###########################################################
# main target
install:: $(LIBS_TO_BUILD:%=%install-libs) \
	install-include install-man install-docs
install:: $(EXEC_TO_BUILD:%=%install-exec) \

# main target
$(INT_BITS)install-libs:
	$(MAKE) "BITS=$(INT_BITS)" "AFLAGS=$(INT_AFLAGS)" install-libs

# main target
$(LONG_BITS)install-libs:
	$(MAKE) "BITS=$(LONG_BITS)" "AFLAGS=$(LONG_AFLAGS)" install-libs

#main target
$(GMP_BITS)install-libs:
	$(MAKE) "BITS=$(GMP_BITS)" "AFLAGS=$(GMP_AFLAGS)" install-libs

install-libs: $(INSTALL_LIB)
install-shared: lib-shared
	$(mkinstalldirs) $(LIBDIR)
	$(INSTALL) $(OBJ_DIR)/$(PSHARED) $(LIBDIR)/
	$(RM) $(LIBDIR)/libpolylib$(BITS).$(SHEXT)
	$(LN_S) $(LIBDIR)/$(PSHARED) $(LIBDIR)/libpolylib$(BITS).$(SHEXT)
	$(LDCONFIG)
install-static: lib-static
	$(mkinstalldirs) $(LIBDIR)
	$(INSTALL) $(OBJ_DIR)/$(PSTATIC) $(LIBDIR)/
	$(RM) $(LIBDIR)/libpolylib$(BITS).a
	$(LN_S) $(LIBDIR)/$(PSTATIC) $(LIBDIR)/libpolylib$(BITS).a

install-include:
	$(mkinstalldirs) $(INCLUDEDIR)/polylib
	$(INSTALL_DATA) ./include/polylib/*.h $(INCLUDEDIR)/polylib

install-man:
# to be done...
install-docs:
	$(mkinstalldirs) $(DOCSDIR)
	$(INSTALL_DATA) doc/* $(DOCSDIR)/
	$(mkinstalldirs) $(DOCSDIR)/examples/ehrhart
	$(INSTALL_DATA) Test/ehrhart/*.in $(DOCSDIR)/examples/ehrhart

# EXEC
# main target
$(INT_BITS)install-exec::
	$(MAKE) "BITS=$(INT_BITS)" "AFLAGS=$(INT_AFLAGS)" install-exec

# main target
$(LONG_BITS)install-exec::
	$(MAKE) "BITS=$(LONG_BITS)" "AFLAGS=$(LONG_AFLAGS)" install-exec

# main target
$(GMP_BITS)install-exec::
	$(MAKE) "BITS=$(GMP_BITS)" "AFLAGS=$(GMP_AFLAGS)" install-exec

install-exec: $(POLY_EXEC)
	$(mkinstalldirs) $(BINDIR)
	$(INSTALL) $(POLY_EXEC) $(BINDIR)

#### Link the executables to INT or to LONG executables with the extra suffix

# main target
$(INT_BITS)ln-exec:
	$(MAKE) "BITS=$(INT_BITS)" "AFLAGS=$(INT_AFLAGS)" ln-exec
# main target
$(LONG_BITS)ln-exec:
	$(MAKE) "BITS=$(LONG_BITS)" "AFLAGS=$(LONG_AFLAGS)" ln-exec
# main target
$(GMP_BITS)ln-exec:
	$(MAKE) "BITS=$(GMP_BITS)" "AFLAGS=$(GMP_AFLAGS)" ln-exec
ln-exec: $(PEXEC:%=ln-%)
$(PEXEC:%=ln-%):
	(cd $(BINDIR); $(LN_S) $(@:ln-%=%$(EXEC_EXTRA_SUFFIX)) $(@:ln-%=%) )

# UNINSTALL-ALL
# main target
uninstall:: uninstall-libs uninstall-exec uninstall-man uninstall-docs
uninstall-libs:
	$(RM) $(LIBDIR)/libpolylib*
	$(RM) -r $(INCLUDEDIR)/polylib
uninstall-exec:
	( cd $(BINDIR) ; $(RM) $(PEXEC:%=%*) )
uninstall-man:

uninstall-docs:
	$(RM) -r $(DOCSDIR)

###########################################################
## Libs : static and shared
###########################################################
libs: $(LIBS_TO_BUILD:%=%libs)

# main target
$(INT_BITS)libs:
	$(MAKE) "BITS=$(INT_BITS)" "AFLAGS=$(INT_AFLAGS)" mlibs

# main target
$(LONG_BITS)libs:
	$(MAKE) "BITS=$(LONG_BITS)" "AFLAGS=$(LONG_AFLAGS)" mlibs

# main target
$(GMP_BITS)libs:
	$(MAKE) "BITS=$(GMP_BITS)" "AFLAGS=$(GMP_AFLAGS)" mlibs


# local targets
mlibs: lib-shared lib-static

lib-shared:: $(OBJ_DIR)
lib-shared:: $(OBJ_DIR)/$(PSHARED)
lib-static:: $(OBJ_DIR)
lib-static:: $(OBJ_DIR)/$(PSTATIC)
	$(RM) $(OBJ_DIR)/libpolylib$(BITS).a
	$(LN_S) $(PSTATIC) $(OBJ_DIR)/libpolylib$(BITS).a

$(OBJ_DIR)/$(PSTATIC): $(PHEADERS) $(LIB_OBJECTS)  
	(cd $(ARITH_DIR) ; $(MAKE) "CFLAGS=$(CFLAGS)" "OBJ_DIR=$(OBJ_DIR)" )
	$(RM) $(OBJ_DIR)/$(PSTATIC)
	cp $(ARITH_DIR)/$(OBJ_DIR)/arithmetique.a $(OBJ_DIR)/$(PSTATIC)
	$(AR) -q $(OBJ_DIR)/$(PSTATIC) $(LIB_OBJECTS)
	@$(RANLIB) $(OBJ_DIR)/$(PSTATIC)

$(OBJ_DIR)/$(PSHARED): $(LIB_OBJECTS)
	(cd $(ARITH_DIR) ; $(MAKE) "CFLAGS=$(CFLAGS)" "OBJ_DIR=$(OBJ_DIR)" )
	$(LD) $(SHAREDLIB_FLAG) -o $(OBJ_DIR)/$(PSHARED) $(LIB_OBJECTS) \
		$(ARITH_DIR)/$(OBJ_DIR)/arithmetique.a

###########################################################
## Cleans
###########################################################
# main target
clean:
	(cd $(ARITH_DIR) ; $(MAKE) "OBJ_DIR=$(OBJ_DIR)" clean)
	$(RM) -r Obj.*
distclean: clean
	$(RM) config.cache config.log config.status

###########################################################
## Tests
###########################################################

# main target
test: $(EXEC_TO_BUILD:%=%test)

$(INT_BITS)test:
	$(MAKE) "BITS=$(INT_BITS)" "AFLAGS=$(INT_AFLAGS)" alltests
$(LONG_BITS)test:
	$(MAKE) "BITS=$(LONG_BITS)" "AFLAGS=$(LONG_AFLAGS)" alltests
$(GMP_BITS)test:
	$(MAKE) "BITS=$(GMP_BITS)" "AFLAGS=$(GMP_AFLAGS)" gmptests

alltests: 
	@echo
	@echo ----------------------------------------------------
	@echo ------------------- $(BITS) BITS TESTS ------------------
	@echo ----------------------------------------------------
	(cd Test ; $(MAKE) \
		"OBJ_DIR=../$(OBJ_DIR)" "EXEC_EXTRA_SUFFIX=$(EXEC_EXTRA_SUFFIX)" )

gmptests: 
	@echo
	@echo ----------------------------------------------------
	@echo --------------GNU MULTI-PRECISION TESTS ------------
	@echo ----------------------------------------------------
	(cd Test ; $(MAKE) \
		"OBJ_DIR=../$(OBJ_DIR)" "EXEC_EXTRA_SUFFIX=$(EXEC_EXTRA_SUFFIX)" )

longtest: $(EXEC_TO_BUILD:%=%longtest)

$(INT_BITS)longtest:
	$(MAKE) "BITS=$(INT_BITS)" "AFLAGS=$(INT_AFLAGS)" alllongtests
$(LONG_BITS)longtest:
	$(MAKE) "BITS=$(LONG_BITS)" "AFLAGS=$(LONG_AFLAGS)" alllongtests
$(GMP_BITS)longtest:
	$(MAKE) "BITS=$(GMP_BITS)" "AFLAGS=$(GMP_AFLAGS)" gmplongtests

alllongtests: 
	@echo
	@echo ----------------------------------------------------
	@echo --------------- $(BITS) BITS LONG TESTS ------------
	@echo ----------------------------------------------------             
	(cd Test ; $(MAKE) "OBJ_DIR=../$(OBJ_DIR)" \
		  "EXEC_EXTRA_SUFFIX=$(EXEC_EXTRA_SUFFIX)" long_tests)
gmplongtests: 
	@echo
	@echo ----------------------------------------------------
	@echo ------------GNU MULTI-PRECISION LONG TESTS ---------
	@echo ----------------------------------------------------            
	(cd Test ; $(MAKE) "OBJ_DIR=../$(OBJ_DIR)" \
		  "EXEC_EXTRA_SUFFIX=$(EXEC_EXTRA_SUFFIX)" long_tests)


########################################################################
##  Lib Objects
######################################################################## 
$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(LIB_OBJECTS): $(OBJ_DIR)/%.o:$(POLYLIB_SRC)/%.c $(PHEADERS)
	$(CC) -c $(CFLAGS) $(POLYLIB_SRC)/$*.c -o $(OBJ_DIR)/$*.o

#$(PHEADERS): $(LIB_CFILES)
#	-mv -f $@ $@.old
#	cextract +a +w72 $(EXTRA_INCLUDES) -H_$*_H_ -o $@ $(@:.h=.c)

########################################################################
## Executables
########################################################################
$(INT_BITS)exec::
	$(MAKE) "BITS=$(INT_BITS)" "AFLAGS=$(INT_AFLAGS)" allexec
$(LONG_BITS)exec::
	$(MAKE) "BITS=$(LONG_BITS)" "AFLAGS=$(LONG_AFLAGS)" allexec
$(GMP_BITS)exec::
	$(MAKE) "BITS=$(GMP_BITS)" "AFLAGS=$(GMP_AFLAGS)" "EXTRA_LIBS=$(EXTRA_LIBS)" allexec

allexec:: $(OBJ_DIR)
allexec:: $(POLY_EXEC)
	@echo "---------------------------------------------------"
	@echo "Successfully built $(BITS) bits executables."
	@echo "---------------------------------------------------"
	@echo "Type 'make test' to test the library"
	@echo "Type 'make longtest' to further test the library (slower)"

# constraints2polyhedron
$(OBJ_DIR)/c2p$(EXEC_EXTRA_SUFFIX): $(OBJ_DIR)/c2p.o $(LIB)
	$(LD) $(LDFLAGS) -o $@ $(OBJ_DIR)/c2p.o \
		$(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)
$(OBJ_DIR)/c2p.o: $(POLYLIB_SRC)/c2p.c $(PHEADERS)
	$(CC) -c $(CFLAGS) $(POLYLIB_SRC)/c2p.c -o $(OBJ_DIR)/c2p.o

# rays2polyhedron
$(OBJ_DIR)/r2p$(EXEC_EXTRA_SUFFIX): $(OBJ_DIR)/r2p.o $(LIB)
	$(LD) $(LDFLAGS) -o $@ $(OBJ_DIR)/r2p.o \
		$(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)
$(OBJ_DIR)/r2p.o: $(POLYLIB_SRC)/r2p.c $(PHEADERS)
	$(CC) -c $(CFLAGS) $(POLYLIB_SRC)/r2p.c -o $(OBJ_DIR)/r2p.o

# find vertices
$(OBJ_DIR)/findv$(EXEC_EXTRA_SUFFIX): $(OBJ_DIR)/findv.o $(LIB)
	$(LD) $(LDFLAGS) -o $@ $(OBJ_DIR)/findv.o \
		$(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)
$(OBJ_DIR)/findv.o: $(POLYLIB_SRC)/findv.c $(PHEADERS)
	$(CC) -c $(CFLAGS) $(POLYLIB_SRC)/findv.c -o $(OBJ_DIR)/findv.o

# find validity domains+vertices
$(OBJ_DIR)/pp$(EXEC_EXTRA_SUFFIX): $(OBJ_DIR)/pp.o $(LIB)
	$(LD) $(LDFLAGS) -o $@ $(OBJ_DIR)/pp.o \
		$(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)
$(OBJ_DIR)/pp.o: $(POLYLIB_SRC)/pp.c $(PHEADERS)
	$(CC) -c $(CFLAGS) $(POLYLIB_SRC)/pp.c -o $(OBJ_DIR)/pp.o


# computes vertices, validity domains and ehrhart polynomial
$(OBJ_DIR)/ehrhart$(EXEC_EXTRA_SUFFIX): $(POLYLIB_SRC)/ehrhart.c \
				$(LIB) $(PHEADERS)
	$(CC) $(LDFLAGS) $(CFLAGS) -DEMAIN -DEP_EVALUATION  -o $@ \
		$(EHRHART_MP:%=$(POLYLIB_SRC)/%) $(POLYLIB_SRC)/ehrhart.c \
		$(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)

# verifies the ehrhart polynomial is correct
$(OBJ_DIR)/verif_ehrhart$(EXEC_EXTRA_SUFFIX): $(POLYLIB_SRC)/verif_ehrhart.c \
				$(LIB) $(PHEADERS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ \
		$(POLYLIB_SRC)/verif_ehrhart.c \
		$(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)

# misc : disjoint and convex union of polyhedra (cf. sources)
$(OBJ_DIR)/union_disjointe$(EXEC_EXTRA_SUFFIX): $(PHEADERS) $(LIB) \
			$(POLYLIB_SRC)/union_disjointe.c
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ \
		$(POLYLIB_SRC)/union_disjointe.c $(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)
$(OBJ_DIR)/union_convex$(EXEC_EXTRA_SUFFIX): $(PHEADERS) $(LIB) \
			$(POLYLIB_SRC)/union_convex.c
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ \
		$(POLYLIB_SRC)/union_convex.c $(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)

# original PolyLib1 tests
$(OBJ_DIR)/testlib$(EXEC_EXTRA_SUFFIX): $(POLYLIB_SRC)/testlib.c $(LIB)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(POLYLIB_SRC)/testlib.c \
		$(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)
$(OBJ_DIR)/polytest$(EXEC_EXTRA_SUFFIX): $(POLYLIB_SRC)/polytest.c $(LIB)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(POLYLIB_SRC)/polytest.c \
		 $(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)

# original Zpolyhedron tests
$(OBJ_DIR)/Zpolytest$(EXEC_EXTRA_SUFFIX): $(POLYLIB_SRC)/Zpolytest.c $(LIB)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(POLYLIB_SRC)/Zpolytest.c \
		 $(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)

$(OBJ_DIR)/example$(EXEC_EXTRA_SUFFIX): $(POLYLIB_SRC)/example.c $(LIB)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(POLYLIB_SRC)/example.c \
		 $(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)

########################################################################
vars.mk: vars.mk.in
	./configure

########################################################################
## END
########################################################################



