# $Id$

# expected macros:
# ROOT
ifndef ROOT
$(error "expected ROOT macro not found!")
endif

# ARCH (or default provided)

# INC_TARGET: header file for directory (on INC_CFILES or LIB_CFILES)

# LIB_CFILES: C files that are included in the library
# LIB_TARGET: generated library file
# BIN_TARGET: generated binary files

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

# the default target is to "recompile" the current directory
all: recompile

recompile: phase0 phase1 phase2 phase3 phase4 phase5 phase6

########################################################################## ROOT

ifndef INSTALL_DIR
INSTALL_DIR	= $(ROOT)
endif # INSTALL_DIR

ifndef PIPS_ROOT
PIPS_ROOT	= $(ROOT)/../pips
endif # PIPS_ROOT

ifndef NEWGEN_ROOT
NEWGEN_ROOT	= $(ROOT)/../newgen
endif # NEWGEN_ROOT

ifndef LINEAR_ROOT
LINEAR_ROOT	= $(ROOT)/../linear
endif # LINEAR_ROOT

ifndef EXTERN_ROOT
EXTERN_ROOT	= $(ROOT)/../extern
endif # EXTERN_ROOT

# where are make files
MAKE.d	= $(ROOT)/makes

########################################################################## ARCH

include $(MAKE.d)/arch.mk

###################################################### INSTALLATION DIRECTORIES

# where to install stuff
BIN.d	= $(INSTALL_DIR)/bin/$(ARCH)
EXE.d	= $(INSTALL_DIR)/bin
LIB.d	= $(INSTALL_DIR)/lib
INC.d	= $(INSTALL_DIR)/include
ETC.d	= $(INSTALL_DIR)/etc
# By default, install the documentation directly into $(DOC.d) but DOC.subd can
# be used to specify a subdirectory:
DOC.d	= $(INSTALL_DIR)/doc
MAN.d	= $(INSTALL_DIR)/man
# By default, install HTML stuff directly into $(HTM.d) but HTM.subd can
# be used to specify a subdirectory:
HTM.d	= $(INSTALL_DIR)/html
UTL.d	= $(INSTALL_DIR)/utils
SHR.d	= $(INSTALL_DIR)/share
RTM.d	= $(INSTALL_DIR)/runtime

# do not include for some targets such as "clean"
clean: NO_INCLUDES=1
export NO_INCLUDES

ifndef NO_INCLUDES

# special definitions for the target architecture
include $(MAKE.d)/$(ARCH).mk

# svn related targets...
include $(MAKE.d)/svn.mk

# site specific stuff...
-include $(MAKE.d)/config.mk

# auto generate config if necessary
$(MAKE.d)/config.mk:
	echo "MAKEFLAGS = -j1" > $@

endif # NO_INCLUDES

# project specific rules are included anyway, as there may be clean stuff.
ifdef PROJECT
include $(MAKE.d)/$(PROJECT).mk
endif # PROJECT

# ??? fix path...
PATH	:= $(PATH):$(NEWGEN_ROOT)/bin:$(NEWGEN_ROOT)/bin/$(ARCH)

###################################################################### DO STUFF

UTC_DATE := $(shell date -u | tr ' ,()' '_')
CPPFLAGS += -DSOFT_ARCH='$(ARCH)' -I$(ROOT)/include

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
PROTOIZE= $(PROTO) $(PRFLAGS) -E "$(PREPROC) -DCPROTO_IS_PROTOTYPING"
M4FLT	= $(M4) $(M4OPT) $(M4FLAGS)
MAKEDEP	= $(CC) $(CMKDEP) $(CANSI) $(CFLAGS) $(COPT) $(CPPOPT) $(CPPFLAGS)
NODIR	= --no-print-directory
COPY	= cp
MOVE	= mv
JAVAC	= javac
JNI	= javah -jni
MKDIR	= mkdir -p -m 755
RMDIR	= rmdir
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
endif

# generation by make recursion
$(DEPEND): $(LIB_CFILES) $(OTHER_CFILES) $(DERIVED_CFILES)
	touch $@
	test -s $(DEPEND) || $(MAKE) depend

# actual generation is done on demand only
depend: $(DERIVED_HEADERS) $(INC_TARGET)
	$(MAKEDEP) $(LIB_CFILES) $(OTHER_CFILES) $(DERIVED_CFILES) | \
	sed \
		-e 's,^\(.*\.o:\),$(ARCH)/\1,;' \
		-e 's,$(subst .,\.,$(PIPS_ROOT)),$$(PIPS_ROOT),g' \
		-e 's,$(subst .,\.,$(LINEAR_ROOT)),$$(LINEAR_ROOT),g' \
		-e 's,$(subst .,\.,$(NEWGEN_ROOT)),$$(NEWGEN_ROOT),g' \
		-e 's,$(subst .,\.,$(ROOT)),$$(ROOT),g' > $(DEPEND)

clean: depend-clean

depend-clean:; $(RM) .depend.*

endif # need_depend

########################################################### CONFIGURATION FILES

ifdef ETC_TARGET

INSTALL_ETC	+= $(ETC_TARGET)

endif # ETC_TARGET

ifdef INSTALL_ETC

$(ETC.d):
	$(MKDIR) $@

# Deal also with directories.
# By the way, how to install directories with "install" ?
.build_etc: $(INSTALL_ETC)
	# no direct dependency on target directory
	$(MAKE) $(ETC.d)
	for f in $(INSTALL_ETC) ; do \
	  if [ -d $$f ] ; then \
	    find $$f -type d -name '.svn' -prune -o -type f -print | \
	      while read file ; do \
	        echo "installing $$file" ; \
		$(INSTALL) -D -m 644 $$file $(ETC.d)/$$file ; \
	      done ; \
	  else \
	    $(CMP) $$f $(ETC.d)/$$f || \
	      $(INSTALL) -m 644 $$f $(ETC.d) ; \
	  fi ; \
	done
	touch $@

clean: etc-clean

etc-clean:
	$(RM) .build_etc

phase1: .build_etc

endif # INSTALL_ETC

####################################################################### HEADERS

ifdef INC_TARGET
# generate directory header file with cproto

ifndef INC_CFILES
INC_CFILES	= $(LIB_CFILES)
endif # INC_CFILES

name	= $(subst -,_, $(notdir $(CURDIR)))

build-header-file:
	$(COPY) $(TARGET)-local.h $(INC_TARGET); \
	{ \
	  echo "/* header file built by $(PROTO) */"; \
	  echo "#ifndef $(name)_header_included";\
	  echo "#define $(name)_header_included";\
	  cat $(TARGET)-local.h;\
	  $(PROTOIZE) $(INC_CFILES) | \
	  sed -f $(MAKE.d)/proto.sed ; \
	  echo "#endif /* $(name)_header_included */"; \
	} > $(INC_TARGET).tmp
	$(MOVE) $(INC_TARGET).tmp $(INC_TARGET)

# force local header construction, but only if really necessary;-)
# the point is that the actual dependency is hold by the ".header" file,
# so we must just check whether this file is up to date.
header:	.header $(INC_TARGET)

# .header carries all dependencies for INC_TARGET:
.header: $(TARGET)-local.h $(DERIVED_HEADERS) $(LIB_CFILES) 
	$(MAKE) $(GMKNODIR) build-header-file
	touch .header

$(INC_TARGET): $(TARGET)-local.h
	$(RM) .header; $(MAKE) $(GMKNODIR) .header

phase2:	$(INC_TARGET)

clean: header-clean

header-clean:
	$(RM) $(INC_TARGET) .header

INSTALL_INC	+=   $(INC_TARGET)

endif # INC_TARGET

ifdef INSTALL_INC

phase2: .build_inc

$(INC.d):; $(MKDIR) $(INC.d)

.build_inc: $(INSTALL_INC)
	# no dep on target dir
	$(MAKE) $(INC.d)
	for f in $(INSTALL_INC) ; do \
	  $(CMP) $$f $(INC.d)/$$f || \
	    $(INSTALL) -m 644 $$f $(INC.d) ; \
	done
	touch $@

clean: inc-clean

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
lib: $(ARCH)/$(LIB_TARGET)

INSTALL_LIB	+=   $(addprefix $(ARCH)/,$(LIB_TARGET))

endif # LIB_TARGET


ifdef INSTALL_LIB

phase2: $(ARCH)/.dir

phase4:	.build_lib.$(ARCH)

$(INSTALL_LIB): $(ARCH)/.dir

$(LIB.d):
	$(MKDIR) $@

$(LIB.d)/$(ARCH): $(LIB.d)
	$(MKDIR) $@

.build_lib.$(ARCH): $(INSTALL_LIB)
	# no dep on target dir
	$(MAKE) $(LIB.d)/$(ARCH)
	for l in $(INSTALL_LIB) ; do \
	  $(CMP) $$l $(LIB.d)/$$l || \
	    $(INSTALL) -m 644 $$l $(LIB.d)/$(ARCH) ; \
	done
	touch $@

clean: lib-clean

lib-clean:; $(RM) $(ARCH)/$(LIB_TARGET) .build_lib.*

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
	$(MAKE) phase6

full-compile: compile
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
phase4:
phase5:
phase6:
phase7:

clean: phase0-clean

phase0-clean:
	$(RM) .build_bootstrap

ifdef INSTALL_EXE

phase2: .build_exe

$(EXE.d):
	$(MKDIR) $@

.build_exe: $(INSTALL_EXE)
	# no direct deps on target dir
	$(MAKE) $(EXE.d)
	$(INSTALL) -m 755 $(INSTALL_EXE) $(EXE.d)
	touch $@

clean: exe-clean

exe-clean:
	$(RM) .build_exe

endif # INSTALL_EXE


# binaries
ifdef BIN_TARGET
INSTALL_BIN	+=   $(addprefix $(ARCH)/,$(BIN_TARGET))
endif # BIN_TARGET

ifdef INSTALL_BIN

phase5: .build_bin.$(ARCH)

$(INSTALL_BIN): $(ARCH)/.dir

$(BIN.d):
	$(MKDIR) $@

.build_bin.$(ARCH): $(INSTALL_BIN)
	# no direct deps on target dir
	$(MAKE) $(BIN.d)
	$(INSTALL) -m 755 $(INSTALL_BIN) $(BIN.d)
	touch $@

clean: bin-clean

bin-clean:
	$(RM) .build_bin.*

endif # INSTALL_BIN

# Documentation
ifdef INSTALL_DOC

phase6: .build_doc

$(DOC.d):; $(MKDIR) $(DOC.d)

# There may be, but not necessarily, a subdirectory...
ifdef DOC.subd
DOC.dest	= $(DOC.d)/$(DOC.subd)
$(DOC.dest): $(DOC.d)
	$(MKDIR) $(DOC.dest)
else # no subdir
DOC.dest	= $(DOC.d)
endif # DOC.subd

.build_doc: $(INSTALL_DOC)
	# no direct deps on target dir
	$(MAKE) $(DOC.dest)
	$(INSTALL) -m 644 $(INSTALL_DOC) $(DOC.dest)
	touch $@

clean: doc-clean

doc-clean:
	$(RM) .build_doc

endif # INSTALL_DOC

# manuel
ifdef INSTALL_MAN

phase6: .build_man

$(MAN.d):; $(MKDIR) $(MAN.d)

.build_man: $(INSTALL_MAN)
	# no direct deps on target dir
	$(MAKE) $(MAN.d)
	$(INSTALL) -m 644 $(INSTALL_MAN) $(MAN.d)
	touch $@

clean: man-clean

man-clean:
	$(RM) .build_man

endif # INSTALL_MAN

# html documentations after everything else...
# Build the documentation only if it is expected and possible:
ifdef INSTALL_HTM
ifdef _HAS_HTLATEX_

phase7: .build_htm

$(HTM.d)/$(HTM.subd):; $(MKDIR) -p $(HTM.d)/$(HTM.subd)

.build_htm: $(INSTALL_HTM)
	# no direct deps on target dir
	$(MAKE) $(HTM.d)/$(HTM.subd)
	# Deal also with directories.
	# By the way, how to install directories with "install" ?
	# The cp -r*f*. is to overide read-only that may exist in the target
	for f in $(INSTALL_HTM) ; do \
	  if [ -d $$f ] ; then \
	    cp -rf $$f $(HTM.d)/$(HTM.subd) ; \
	else \
	    $(CMP) $$f $(HTM.d)/$(HTM.subd)/$$f || \
	      $(INSTALL) -m 644 $$f $(HTM.d)/$(HTM.subd) ; \
	  fi ; \
	done
	touch $@

endif # _HAS_HTLATEX_

clean: htm-clean

htm-clean:
	$(RM) .build_htm

endif # INSTALL_HTM

# shared
ifdef INSTALL_SHR

phase2: .build_shr

$(SHR.d):; $(MKDIR) $(SHR.d)

.build_shr: $(INSTALL_SHR)
	# no direct deps on target dir
	$(MAKE) $(SHR.d)
	$(INSTALL) -m 644 $(INSTALL_SHR) $(SHR.d)
	touch $@

clean: shr-clean

shr-clean:
	$(RM) .build_shr

endif # INSTALL_SHR

# utils
ifdef INSTALL_UTL

phase2: .build_utl

$(UTL.d):; $(MKDIR) $(UTL.d)

.build_utl: $(INSTALL_UTL)
	# no direct deps on target dir
	$(MAKE) $(UTL.d)
	$(INSTALL) -m 755 $(INSTALL_UTL) $(UTL.d)
	touch $@

clean: utl-clean

utl-clean:
	$(RM) .build_utl

endif # INSTALL_UTL

# other targets

clean: main-clean

main-clean:
	$(RM) *~ *.tmp

################################################################### DEVELOPMENT

# try development directory under setup_pips.sh
DEVDIR	= $(ROOT)/../../$(PROJECT)_dev

# can be overriden... for instance there are 2 'pipsmake' directories
NEW_BRANCH_NAME	= $(notdir $(CURDIR))

# the old pips development target
install: 
	@echo "See 'create-branch' target to create a development branch"
	@echo "See 'install-branch' target to install a development branch"

# should be ok
force-create-branch: 
	$(MAKE) BRANCH_FLAGS+=--commit create-branch
	-test -d $(DEVDIR)/.svn && $(SVN) update $(DEVDIR)

ifdef SVN_USERNAME
devsubdir	= $(SVN_USERNAME)/
else
devsubdir	= $(USER)/
endif

# create a new private branch
create-branch:
	-@if $(IS_SVN_WC) ; then \
	  if $(IS_BRANCH) . ; then \
	    echo "should not create a branch on a branch?!" ; \
	  else \
	    if test -d $(DEVDIR)/.svn ; then \
		branch=$(DEVDIR)/$(NEW_BRANCH_NAME) ; \
	    else \
		branch=branches/$(devsubdir)$(NEW_BRANCH_NAME) ; \
	    fi ; \
	    $(BRANCH) create $(BRANCH_FLAGS) . $$branch ; \
	  fi ; \
	else \
	  echo "cannot create branch, not a wcpath" ; \
	fi

# hum...
force-install-branch: 
	$(MAKE) BRANCH_FLAGS+=--commit install-branch
	-test -d $(ROOT)/.svn && $(SVN) update $(ROOT)

# install the branch into trunk (production version)
install-branch:
	-@if $(IS_SVN_WC) ; then \
	  if $(IS_BRANCH) . ; then \
	    echo "installing current directory..." ; \
	    $(BRANCH) push $(BRANCH_FLAGS) . ; \
	  else \
	    echo "cannot install current directory, not a branch" ; \
	  fi ; \
	else \
	  echo "cannot install current directory, not under svn" ; \
	fi

remove-branch:
	-@if $(IS_SVN_WC) ; then \
	  if $(IS_BRANCH) . ; then \
	    echo "removing current branch..." ; \
	    svn rm . ; \
	    echo "please commit .. if you agree" ; \
	  else \
	    echo "cannot remove branch, not a branch" ; \
	  fi ; \
	else \
	  echo "cannot remove branch, not under svn" ; \
	fi

branch-diff:
	-@$(IS_BRANCH) . && $(BRANCH) diff

branch-info:
	-@$(IS_BRANCH) . && $(BRANCH) info

branch-avail:	
	-@$(IS_BRANCH) . && $(BRANCH) avail
