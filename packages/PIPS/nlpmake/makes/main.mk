# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

# expected macros:
# ROOT
ifndef ROOT
$(error "expected ROOT macro not found!")
endif

# ARCH (or default provided)

# INC_TARGET: header file for directory (build from INC_CFILES or LIB_CFILES)

# LIB_CFILES: C files that are used in the library
# INC_CFILES : the source really used to generate the INC_TARGET from cproto.
#              If not defined, it is initialized from LIB_CFILES
# LIB_TARGET: generated library file
# BIN_TARGET: generated binary files from the library
# OTHER_CFILES : sources used to build the BIN_TARGET with the library too

# files to be installed in subdirectories:
# INSTALL_INC: headers to be installed (INC_TARGET not included)
# INSTALL_LIB: libraries to be added (LIB_TARGET not included)
# INSTALL_BIN: added binaries (BIN_TARGET not included)
# INSTALL_EXE: added executable (shell scripts or the like)
# INSTALL_ETC: configuration files
# INSTALL_SHR: shared (cross platform) files
# INSTALL_UTL: script utilities
# INSTALL_RTM: runtime-related stuff
# INSTALL_MAN DOC HTM: various documentations

# the default target is to simply "compile" the current directory
all: compile

recompile:
	$(MAKE) clean
	$(MAKE) compile

################################################################### ROOT & ARCH

MAKE.d	= $(ROOT)/makes

include $(MAKE.d)/root.mk
include $(MAKE.d)/arch.mk

###################################################### INSTALLATION DIRECTORIES

ifndef INSTALL_DIR
INSTALL_DIR	= $(ROOT)
endif # INSTALL_DIR

# where to install stuff
BIN.d	= $(INSTALL_DIR)/bin/$(ARCH)
EXE.d	= $(INSTALL_DIR)/bin
LIB.d	= $(INSTALL_DIR)/lib/$(ARCH)
INC.d	= $(INSTALL_DIR)/include
ETC.d	= $(INSTALL_DIR)/etc
PY.d	= $(LIB.d)
# By default, install the documentation directly into $(DOC.d) but DOC.subd can
# be used to specify a subdirectory:
DOC.d	= $(INSTALL_DIR)/doc
MAN.d	= $(INSTALL_DIR)/man
# By default, install HTML stuff directly into $(HTM.d) but HTM.subd can
# be used to specify a subdirectory. It is put in doc because it helps
# with the publication of the html documentation.
HTM.d	= $(INSTALL_DIR)/doc
UTL.d	= $(INSTALL_DIR)/utils
SHR.d	= $(INSTALL_DIR)/share
RTM.d	= $(INSTALL_DIR)/runtime

# do not include for some targets such as "clean"
clean: NO_INCLUDES=1
export NO_INCLUDES

ifndef NO_INCLUDES

# site specific stuff...
ifndef CONFIG_DONE
-include $(MAKE.d)/config.mk
export CONFIG_DONE=1
endif

# special definitions for the target architecture
include $(MAKE.d)/$(ARCH).mk

# svn related targets...
include $(MAKE.d)/svn.mk


# auto generate config if necessary
$(MAKE.d)/config.mk:
	touch $@

endif # NO_INCLUDES

# project specific rules are included anyway, as there may be clean stuff.
ifdef PROJECT
include $(MAKE.d)/$(PROJECT).mk
endif # PROJECT

# ??? fix path...
PATH	:= $(PATH):$(NEWGEN_ROOT)/bin:$(NEWGEN_ROOT)/bin/$(ARCH)

###################################################################### DO STUFF

UTC_DATE := "$(shell date -u)"
CPPFLAGS += -DSOFT_ARCH='$(ARCH)' -I$(INC.d)

ifdef WITH_DYNAMIC_LIBRARIES
CFLAGS+=-fPIC
endif

# {C,CPP,LD,L,Y}OPT macros allow to *add* things from the command line
# as gmake CPPOPT="-DFOO=bar" ... that will be added to the defaults
# a typical interesting example is to put -pg in {C,LD}OPT
#
PREPROC	= $(CC) -E $(CANSI) $(CPPOPT) $(CPPFLAGS)
COMPILE	= $(CC) $(CANSI) $(CFLAGS) $(COPT) $(CPPOPT) $(CPPFLAGS) -c
F77CMP	= $(FC) $(FFLAGS) $(FOPT) -c
LINK	= $(LD) $(LDFLAGS) $(LDOPT) -o
SCAN	= $(LEX) $(LFLAGS) $(LOPT) -t
TYPECK	= $(LINT) $(LINTFLAGS) $(CPPFLAGS) $(LINT_LIBS)
PARSE	= $(YACC) $(YFLAGS) $(YOPT)
ARCHIVE = $(AR) $(ARFLAGS)
PROTOIZE= $(PROTO) $(PRFLAGS) -E "$(PROTO_CPP) $(CPPOPT) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING"
M4FLT	= $(M4) $(M4OPT) $(M4FLAGS)
MAKEDEP	= $(CC) $(CMKDEP) $(CANSI) $(CFLAGS) $(COPT) $(CPPOPT) $(CPPFLAGS)
NODIR	= --no-print-directory
COPY	= cp
MOVE	= mv
JAVAC	= javac
JNI	= javah -jni
MKDIR	= mkdir -p -m 755
RMDIR	= rmdir -p
INSTALL	= install
CMP	= cmp -s

# misc filters
FLTVERB	= sed -f $(MAKE.d)/verbatim.sed
UPPER	= tr '[a-z]' '[A-Z]'

# for easy debugging... e.g. gmake ECHO='something' echo
echo:; @echo $(ECHO)

%: %.c
%: %.o
# Add global file path to help compile mode from editors such as Emacs to
# be clickable:
$(ARCH)/%.o: %.c; $(COMPILE) `pwd`/$< -o $@
$(ARCH)/%.o: %.f; $(F77CMP) `pwd`/$< -o $@

%.class: %.java; $(JAVAC) $<
%.h: %.class; $(JNI) $*

%.f: %.m4f;	$(M4FLT) $(M4FOPT) $< > $@
%.c: %.m4c;	$(M4FLT) $(M4COPT) $< > $@
%.h: %.m4h;	$(M4FLT) $(M4HOPT) $< > $@

# To debug weird macro behaviour, add "file+cpp" target to build a file
# that is preprocessed by the C preprocessor. For example "make blah.c+cpp"
# builds a preprocessed version of blah.c.
#
# Options specific to gcc: keep macro definitions in the file too, keep
# comments, display the list of include files.
%+cpp: %
	$(PREPROC) -dD -CC -H $< > $@


# A rule to debug Makefile forwarding used by forward.mk.
# Do not warn downwards:
debug_forward_makefile:


################################################################## DEPENDENCIES

ifdef LIB_CFILES
need_depend	= 1
endif # LIB_CFILES

ifdef OTHER_CFILES
need_depend	= 1
endif # OTHER_CFILES

ifdef need_depend

DEPEND	= .depend.$(ARCH)

phase0: $(DEPEND)

ifndef NO_INCLUDES
-include $(DEPEND)
endif # NO_INCLUDES

# generation by make recursion
$(DEPEND): $(DERIVED_CFILES)
	touch $@
	test -s $(DEPEND) || $(MAKE) depend

# actual generation is done on demand only
depend: $(DERIVED_HEADERS) $(INC_TARGET)
	$(MAKEDEP) $(LIB_CFILES) $(OTHER_CFILES) $(DERIVED_CFILES) | \
	sed \
		-e 's,^\(.*\.o:\),$(ARCH)/\1,;' \
		-e 's,$(subst .,\.,$(ROOT)),$$(ROOT),g' \
		-e 's,$(subst .,\.,$(LINEAR_ROOT)),$$(LINEAR_ROOT),g' \
		-e 's,$(subst .,\.,$(NEWGEN_ROOT)),$$(NEWGEN_ROOT),g' \
		-e 's,$(subst .,\.,$(PIPS_ROOT)),$$(PIPS_ROOT),g' > $(DEPEND)

clean: depend-clean


depend-clean:; $(RM) .depend.*

endif # need_depend

########################################################### CONFIGURATION FILES

ifdef ETC_TARGET

INSTALL_ETC	+= $(ETC_TARGET)

endif # ETC_TARGET

ifdef INSTALL_ETC

install:etc-install
# Deal also with directories.
# By the way, how to install directories with "install" ?
etc-install .build_etc: $(INSTALL_ETC)
	# no direct dependency on target directory
	$(INSTALL) -d $(ETC.d)
	for f in $(INSTALL_ETC) ; do \
	  if [ -d $$f ] ; then \
	    find $$f -type d -name '.svn' -prune -o -type d -print | \
	      while read dir ; do \
		[ -d $(ETC.d)/$$dir ] || mkdir $(ETC.d)/$$dir ; \
	      done ; \
	    find $$f -type d -name '.svn' -prune -o -type f -print | \
	      while read file ; do \
	        echo "installing $$file" ; \
		$(INSTALL) -m 644 $$file $(ETC.d)/$$file ; \
	      done ; \
	  else \
	    $(CMP) $$f $(ETC.d)/$$f || \
	      $(INSTALL) -m 644 $$f $(ETC.d) ; \
	  fi ; \
	done
	`echo $@ | grep -q install` || touch $@

clean: etc-clean

unbuild: etc-unbuild

etc-unbuild:
	for f in $(INSTALL_ETC) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(ETC.d)/$$f ;\
	done
	-test ! -d $(ETC.d) || $(RMDIR) --ignore-fail-on-non-empty $(ETC.d)

etc-clean:
	$(RM) .build_etc

phase1: .build_etc

endif # INSTALL_ETC

########################################################### PYTHON FILES

ifdef PY_TARGET

INSTALL_PY	+= $(PY_TARGET)

$(PY_TARGET):$(ARCH)

endif # PY_TARGET

ifdef INSTALL_PY

install: py-install
# Deal also with directories.
# By the way, how to install directories with "install" ?
py-install .build_py: $(INSTALL_PY)
	# no direct dependency on target directory
	$(INSTALL) -d $(PY.d)
	for f in $(INSTALL_PY) ; do \
	    $(CMP) $$f $(PY.d)/$$f || \
	      $(INSTALL) -m 644 $$f $(PY.d) ; \
	done
	`echo $@ | grep -q install` || touch $@

clean: py-clean

unbuild: py-unbuild

py-unbuild:
	for f in $(INSTALL_PY) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(PY.d)/$$f ;\
	done
	-test ! -d $(PY.d) || $(RMDIR) --ignore-fail-on-non-empty $(PY.d)

py-clean:
	$(RM) .build_py

phase1: .build_py

endif # INSTALL_PY

####################################################################### HEADERS

ifdef INC_TARGET
# generate directory header file with cproto

ifndef INC_CFILES
INC_CFILES	= $(LIB_CFILES)
endif # INC_CFILES

name	= $(subst -,_, $(notdir $(CURDIR)))

$(INC_TARGET).tmp:
	$(COPY) $(TARGET)-local.h $(INC_TARGET);
	{ \
	  echo "/* Warning! Do not modify this file that is automatically generated! */"; \
	  echo "/* Modify src/Libs/$(TARGET)/$(TARGET)-local.h instead, to add your own modifications. */"; \
	  echo ""; \
	  echo "/* header file built by $(PROTO) */"; \
	  echo ""; \
	  echo "#ifndef $(name)_header_included";\
	  echo "#define $(name)_header_included";\
	  cat $(TARGET)-local.h;\
	  $(PROTOIZE) $(INC_CFILES) $(INC_YFILES) | \
	  sed -f $(MAKE.d)/proto.sed ; \
	  echo "#endif /* $(name)_header_included */"; \
	} > $@ ;

build-header-file: $(INC_TARGET).tmp
	$(MOVE) $< $(INC_TARGET)

.PHONY: build-header-file

# force local header construction, but only if really necessary;-)
# the point is that the actual dependency is hold by the ".header" file,
# so we must just check whether this file is up to date.
header:	.header $(INC_TARGET)

.PHONY: header

# .header carries all dependencies for INC_TARGET:
.header: $(TARGET)-local.h $(DERIVED_HEADERS) $(LIB_CFILES)
	$(MAKE) $(GMKNODIR) build-header-file
	touch .header

$(INC_TARGET): $(TARGET)-local.h
	$(RM) .header; $(MAKE) $(GMKNODIR) .header

# put the -local file in include so that cproto will not complain to much.
phase0: install-temporary-header
.PHONY: install-temporary-header
install-temporary-header:
	$(INSTALL) -d $(INC.d)
	test -f $(INC.d)/$(TARGET).h || { \
	  test -f $(TARGET)-local.h && \
	  cp $(TARGET)-local.h $(INC.d)/$(TARGET).h ; }

phase2:	$(INC_TARGET)

clean: header-clean

.PHONY: header-clean
header-clean:
	$(RM) $(INC_TARGET) .header

INSTALL_INC	+=   $(INC_TARGET)

endif # INC_TARGET

ifdef INSTALL_INC

phase2: .build_inc

install:inc-install
inc-install .build_inc: $(INSTALL_INC)
	# no dep on target dir
	$(INSTALL) -d $(INC.d)
	for f in $(INSTALL_INC) ; do \
	  $(CMP) $$f $(INC.d)/$$f || \
	$(INSTALL) -m 644 $$f $(INC.d) ; \
	done
	`echo $@ | grep -q install` || touch $@

clean: inc-clean
unbuild: inc-unbuild

inc-unbuild:
	for f in $(INSTALL_INC) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(INC.d)/$$f ;\
	done
	-test ! -d $(INC.d) || $(RMDIR) --ignore-fail-on-non-empty $(INC.d)

inc-clean:
	$(RM) .build_inc

endif # INSTALL_INC

####################################################################### LIBRARY

# ARCH subdirectory
$(ARCH):
	test -d $(ARCH) || $(MKDIR) $(ARCH)

# indirectly creates the architecture directory
$(ARCH)/.dir:
	test -d $(ARCH) || $(MAKE) $(ARCH)
	touch $@

clean: arch-clean

arch-clean:
	-test -d $(ARCH) && $(RM) -r $(ARCH)

ifdef LIB_CFILES
ifndef LIB_OBJECTS
LIB_OBJECTS = $(addprefix $(ARCH)/,$(LIB_CFILES:%.c=%.o))
endif # LIB_OBJECTS
endif # LIB_CFILES


ifdef LIB_TARGET

$(ARCH)/$(LIB_TARGET): $(LIB_OBJECTS)
	$(ARCHIVE) $(ARCH)/$(LIB_TARGET) $(LIB_OBJECTS)
	ranlib $@

# indirect dependency to trigger the mkdir without triggering a full rebuild
# $(ARCH) directory must exist, but its date does not matter
# is there a better way?
$(LIB_OBJECTS): $(ARCH)/.dir

# alias for FI
lib: $(LIB_TARGET)

INSTALL_LIB	+=   $(LIB_TARGET)

endif # LIB_TARGET

ifdef DYNLIB_TARGET

phase5:.build_dynlib.$(ARCH)

ifdef DYNLIB_OBJECTS
.build_dynlib.$(ARCH) $(ARCH)/$(DYNLIB_TARGET):$(DYNLIB_OBJECTS)
			MAJOR_VERSION="`echo '$(VERSION)' | cut -d '.' -f 1`";\
	        $(LD) -o $@ -shared $(DYNLIB_OBJECTS) $(LDFLAGS) $(LDOPT) -fPIC -Wl,-soname,$(DYNLIB_TARGET).$$MAJOR_VERSION 
endif

ifdef DYNLIB_ARCHIVES
.build_dynlib.$(ARCH) $(ARCH)/$(DYNLIB_TARGET):$(DYNLIB_ARCHIVES)
			MAJOR_VERSION="`echo '$(VERSION)' | cut -d '.' -f 1`";\
	        $(LD) -o $@ -shared -Wl,--whole-archive $(DYNLIB_ARCHIVES) -Wl,--no-whole-archive $(LDFLAGS) $(LDOPT) -fPIC -Wl,-soname,$(DYNLIB_TARGET).$$MAJOR_VERSION 
endif


INSTALL_DYNLIB     +=   $(DYNLIB_TARGET) 

phase2: $(ARCH)/.dir

phase4::	.build_dynlib.$(ARCH)

$(INSTALL_DYNLIB): $(ARCH)/.dir

install:dynlib-install
dynlib-install .build_dynlib.$(ARCH): $(ARCH)/$(INSTALL_DYNLIB)
	# no dep on target dir
	$(INSTALL) -d $(LIB.d)
	MAJOR_VERSION="`echo '$(VERSION)' | cut -d '.' -f 1`";\
	for l in $(INSTALL_DYNLIB) ; do \
	  $(CMP) $(ARCH)/$$l $(LIB.d)/$$l.$$MAJOR_VERSION || \
	    $(INSTALL) -m 644 $(ARCH)/$$l $(LIB.d)/$$l.$$MAJOR_VERSION ; \
		$(RM) $(LIB.d)/$$l ;\
		cd $(LIB.d) && ln -s $$l.$$MAJOR_VERSION $$l && cd -;\
	done
	`echo $@ | grep -q install` || touch $@

clean: dynlib-clean

dynlib-clean:; $(RM) $(ARCH)/$(DYNLIB_TARGET) .build_dynlib.*

unbuild: dynlib-unbuild

dynlib-unbuild:
	for f in $(INSTALL_DYNLIB) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM)  $(LIB.d)/$$f ;\
		echo "uninstalling $$f.*" ; \
	  	$(RM)  $(LIB.d)/$$f.* ;\
	done
	-test ! -d $(LIB.d) || $(RMDIR) --ignore-fail-on-non-empty $(LIB.d)

endif # DYNLIB_TARGET

ifdef INSTALL_LIB

phase2: $(ARCH)/.dir

phase4::	.build_lib.$(ARCH)

$(INSTALL_LIB): $(ARCH)/.dir

install:lib-install
lib-install .build_lib.$(ARCH): $(addprefix $(ARCH)/,$(INSTALL_LIB))
	# no dep on target dir
	$(INSTALL) -d $(LIB.d)
	for l in $(addprefix $(ARCH)/,$(INSTALL_LIB)) ; do \
	  $(CMP) $$l $(LIB.d)/$$l || \
	    $(INSTALL) -m 644 $$l $(LIB.d) ; \
	done
	`echo $@ | grep -q install` || touch $@

clean: lib-clean

lib-clean:; $(RM) $(ARCH)/$(LIB_TARGET) .build_lib.*

unbuild: lib-unbuild

lib-unbuild:
	for f in $(INSTALL_LIB) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(LIB.d)/$$f ;\
	done
	-test ! -d $(LIB.d) || $(RMDIR) --ignore-fail-on-non-empty $(LIB.d)

endif # INSTALL_LIB

######################################################################## PHASES

# multiphase compilation?

compile:
	$(MAKE) phase0
	$(MAKE) phase1
	$(MAKE) phase2
	$(MAKE) phase3
	$(MAKE) phase4
	$(MAKE) phase5

doc-compile: compile
	$(MAKE) phase6

full-compile: doc-compile
	$(MAKE) phase7

#install: recompile

# empty dependencies to please compile targets
phase0: .build_bootstrap

.build_bootstrap:
	@echo
	@echo "Bootstrap the .h header files."
	@echo "It displays a lot of error... but it should be normal."
	@echo "Next time in phase3 we should reach a fix point"
	@echo
	touch $@

phase1:
phase2:
phase3:
phase4::
phase5:
phase6:
phase7:

.PHONY: phase0 phase1 phase2 phase3 phase4 phase5 phase6 phase7

clean: phase0-clean
phase0-clean:
	$(RM) .build_bootstrap

ifdef INSTALL_EXE

phase2: .build_exe

install:exe-install
exe-install .build_exe: $(INSTALL_EXE)
	# no direct deps on target dir
	$(INSTALL) -d $(EXE.d)
	$(INSTALL) -m 755 $(INSTALL_EXE) $(EXE.d)
	`echo $@ | grep -q install` || touch $@

clean: exe-clean

exe-clean:
	$(RM) .build_exe

unbuild: exe-unbuild
exe-unbuild:
	for f in $(INSTALL_EXE) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(EXE.d)/$$f ;\
	done
	-test ! -d $(EXE.d) || $(RMDIR) --ignore-fail-on-non-empty $(EXE.d)


endif # INSTALL_EXE


# binaries
ifdef BIN_TARGET
INSTALL_BIN	+=   $(addprefix $(ARCH)/,$(BIN_TARGET))
endif # BIN_TARGET

ifdef INSTALL_BIN

phase5: .build_bin.$(ARCH)

$(INSTALL_BIN): $(ARCH)/.dir

install:bin-install
bin-install .build_bin.$(ARCH): $(INSTALL_BIN)
	# no direct deps on target dir
	$(INSTALL) -d $(BIN.d)
	$(INSTALL) -m 755 $(INSTALL_BIN) $(BIN.d)
	`echo $@ | grep -q install` || touch $@

clean: bin-clean

bin-clean:
	$(RM) .build_bin.*

unbuild: bin-unbuild

bin-unbuild:
	BIN="`echo $(BIN.d) | sed -e 's,/$(ARCH),,'`" ;\
	for f in $(INSTALL_BIN) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $${BIN}/$$f ;\
		F=`echo $$f | sed -e 's,$(ARCH)/,,'` ;\
		echo "uninstalling $$F" ; \
		$(RM) -r $${BIN}/$$F ;\
	done
	-test ! -d $(BIND.d) || $(RMDIR) --ignore-fail-on-non-empty $(BIN.d)
	-BIN="`echo $(BIN.d) | sed -e 's,/$(ARCH),,'`" ;\
	 test ! -d $${BIN} || $(RMDIR) --ignore-fail-on-non-empty $${BIN}

endif # INSTALL_BIN

# Documentation
ifdef INSTALL_DOC

phase6: .build_doc

# There may be, but not necessarily, a subdirectory...
ifdef DOC.subd
DOC.dest	= $(DOC.d)/$(DOC.subd)
else # no subdir
DOC.dest	= $(DOC.d)
endif # DOC.subd

install:doc-install
doc-install .build_doc: $(INSTALL_DOC)
	# no direct deps on target dir
	$(INSTALL) -d $(DOC.dest)
	$(INSTALL) -m 644 $(INSTALL_DOC) $(DOC.dest)
	`echo $@ | grep -q install` || touch $@

clean: doc-clean

doc-clean:
	$(RM) .build_doc

unbuild: doc-unbuild

doc-unbuild:
	for f in $(INSTALL_DOC) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(DOC.dest)/$$f ;\
	done
	-test ! -d $(DOC.dest) || $(RMDIR) --ignore-fail-on-non-empty $(DOC.dest)
	-test ! -d $(DOC.d) || $(RMDIR) --ignore-fail-on-non-empty $(DOC.d)
	-test ! -d $(DOC.d) || $(RMDIR) --ignore-fail-on-non-empty $(DOC.d)

endif # INSTALL_DOC

# manuel
ifdef INSTALL_MAN

phase6: .build_man

install:man-install
man-install .build_man: $(INSTALL_MAN)
	# no direct deps on target dir
	$(INSTALL) -d $(MAN.d)
	$(INSTALL) -m 644 $(INSTALL_MAN) $(MAN.d)
	`echo $@ | grep -q install` || touch $@

clean: man-clean

man-clean:
	$(RM) .build_man

unbuild: man-unbuild

man-unbuild:
	for f in $(INSTALL_MAN) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(MAN.d)/$$f ;\
	done
	-test ! -d $(MAN.d) || $(RMDIR) --ignore-fail-on-non-empty $(MAN.d)

endif # INSTALL_MAN

# html documentations after everything else...
# Build the documentation only if it is expected and possible:
ifdef INSTALL_HTM
ifdef _HAS_HTLATEX_

phase7: .build_htm

install:htm-install
htm-install .build_htm: $(INSTALL_HTM)
	# no direct deps on target dir
	$(INSTALL) -d $(HTM.d)/$(HTM.subd)
	# Deal also with directories.
	# By the way, how to install directories with "install" ?
	# The cp -r*f*. is to overide read-only that may exist in the target
	# bad idea: this also copies .svn directories...
	for f in $(INSTALL_HTM) ; do \
	  if [ -d $$f ] ; then \
	    cp -rf $$f $(HTM.d)/$(HTM.subd) ; \
	else \
	    $(CMP) $$f $(HTM.d)/$(HTM.subd)/$$f || \
	      $(INSTALL) -m 644 $$f $(HTM.d)/$(HTM.subd) ; \
	  fi ; \
	done
	`echo $@ | grep -q install` || touch $@

endif # _HAS_HTLATEX_

clean: htm-clean

htm-clean:
	$(RM) .build_htm

unbuild: htm-unbuild

htm-unbuild:
	for f in $(INSTALL_HTM) ; do \
		echo "uninstalling $$f" ; \
		$(RM) -r $(HTM.d)/$(HTM.subd)/$$f ; \
	done
	-test ! -d $(HTM.d)/$(HTM.subd) || $(RMDIR) --ignore-fail-on-non-empty $(HTM.d)/$(HTM.subd)
	-test ! -d $(HTM.d) || $(RMDIR) --ignore-fail-on-non-empty $(HTM.d)

endif # INSTALL_HTM

# shared
ifdef INSTALL_SHR

phase2: .build_shr

install:shr-install
shr-install .build_shr: $(INSTALL_SHR)
	# no direct deps on target dir
	$(INSTALL) -d $(SHR.d)
	$(INSTALL) -m 644 $(INSTALL_SHR) $(SHR.d)
	`echo $@ | grep -q install` || touch $@

clean: shr-clean

shr-clean:
	$(RM) .build_shr

unbuild: shr-unbuild

shr-unbuild:
	for f in $(INSTALL_SHR) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(SHR.d)/$$f ;\
	done
	-test ! -d $(SHR.d) || $(RMDIR) --ignore-fail-on-non-empty $(SHR.d)

endif # INSTALL_SHR

# utils
ifdef INSTALL_UTL

phase2: .build_utl

install:utl-install
utl-install .build_utl: $(INSTALL_UTL)
	# no direct deps on target dir
	$(INSTALL) -d $(UTL.d)
	$(INSTALL) -m 755 $(INSTALL_UTL) $(UTL.d)
	`echo $@ | grep -q install` || touch $@

clean: utl-clean

utl-clean:
	$(RM) .build_utl

unbuild: utl-unbuild

utl-unbuild:
	for f in $(INSTALL_UTL) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(UTL.d)/$$f ;\
	done
	-test ! -d $(UTL.d) || $(RMDIR) --ignore-fail-on-non-empty $(UTL.d)

endif # INSTALL_UTL

# other targets

clean: main-clean

main-clean:
	$(RM) *~ *.tmp

unbuild: main-unbuild

main-unbuild:
	for f in $(INSTALL_RTM) ; do \
		echo "uninstalling $$f" ; \
	  	$(RM) -r $(RTM.d)/$$f ;\
	done
	-test ! -d $(RTM.d) || $(RMDIR) --ignore-fail-on-non-empty $(RTM.d)


# Doxygen documentation:

# By default, doxygen stuff has nothing to do... Rely on explicit Makefile where
# it is really needed (that includes doxygen.mk for example).  If there is
# a doxygen directory somewhere, explicit forward the make inside it:
doxygen doxygen-plain doxygen-graph doxygen-publish::
	if [ -d $@ ]; then $(MAKE) --directory=$@ $@; fi

