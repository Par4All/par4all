# $Id$

# expected macros:
# ROOT
# ARCH (or default provided)

# INC_TARGET: header file for directory (on INC_CFILES or LIB_CFILES)

# LIB_CFILES: C files that are included in the library
# LIB_TARGET: generated library file

# BIN_TARGET: generated binary files

# INSTALL_INC: headers to be installed (INC_TARGET not included)
# INSTALL_LIB: libraries to be added (LIB_TARGET not included)
# INSTALL_BIN: added binaries (BIN_TARGET not included)

all: recompile

recompile: phase0 phase1 phase2 phase3 phase4 phase5

########################################################################## ROOT

ifdef ROOT
INSTALL_DIR	= $(ROOT)
else
$(error "no root directory!")
endif # ROOT

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

# where to install stuff
BIN.d	= $(INSTALL_DIR)/Bin
LIB.d	= $(INSTALL_DIR)/Lib
INC.d	= $(INSTALL_DIR)/Include
DOC.d	= $(INSTALL_DIR)/Doc
HTM.d	= $(INSTALL_DIR)/Html
UTL.d	= $(INSTALL_DIR)/Utils
SHR.d	= $(INSTALL_DIR)/Bin
RTM.d	= $(INSTALL_DIR)/Runtime
MAKE.d	= $(INSTALL_DIR)/makes

########################################################################## ARCH

ifndef ARCH
ifdef PIPS_ARCH
ARCH	= $(PIPS_ARCH)
else
ifdef NEWGEN_ARCH
ARCH	= $(NEWGEN_ARCH)
else
ifdef LINEAR_ARCH
ARCH	= $(LINEAR_ARCH)
else
ARCH	= $(shell $(MAKE.d)/arch.sh)
endif # LINEAR_ARCH
endif # NEWGEN_ARCH
endif # PIPS_ARCH
endif # ARCH

ifndef ARCH
$(error "ARCH macro is not defined")
endif

include $(MAKE.d)/$(ARCH).mk
include $(MAKE.d)/svn.mk
#include $(MAKE.d)/local.mk

###################################################################### DO STUFF

all:

UTC_DATE = $(shell date -u | tr ' ' '_')
CPPFLAGS += -DSOFT_ARCH='"$(ARCH)"' -I$(ROOT)/Include

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
MKDIR	= mkdir -p
RMDIR	= rmdir
INSTALL	= install

# for easy debugging... e.g. gmake ECHO='something' echo
echo:; @echo $(ECHO)

$(ARCH)/%.o: %.c; $(COMPILE) $< -o $@
$(ARCH)/%.o: %.f; $(F77CMP) $< -o $@

%.class: %.java; $(JAVAC) $<
%.h: %.class; $(JNI) $*

%.f: %.m4f;	$(M4FLT) $(M4FOPT) $< > $@
%.c: %.m4c;	$(M4FLT) $(M4COPT) $< > $@
%.h: %.m4h;	$(M4FLT) $(M4HOPT) $< > $@

# latex
%.dvi: %.tex
	-grep '\\makeindex' $*.tex && touch $*.ind
	$(LATEX) $<
	-grep '\\bibdata{' \*.aux && { $(BIBTEX) $* ; $(LATEX) $< ;}
	test ! -f $*.idx || { $(MAKEIDX) $*.idx ; $(LATEX) $< ;}
	$(LATEX) $<
	touch $@

%.newgen: %.tex
	$(RM) $@
	remove-latex-comments $<
	chmod a-w $@

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

-include $(DEPEND)

# generation by make recursion
$(DEPEND): $(LIB_CFILES) $(OTHER_CFILES) $(DERIVED_LIB_HEADERS)
	touch $@
	test -s $(DEPEND) || $(MAKE) depend

# actual generation is done on demand only
depend:
	$(MAKEDEP) $(LIB_CFILES) $(OTHER_CFILES) | \
		sed 's,^\(.*\.o:\),$(ARCH)/\1,' > $(DEPEND)

clean: depend-clean

depend-clean:; $(RM) .depend.*

endif # need_depend

####################################################################### HEADERS

ifdef INC_TARGET
# generate directory header file with cproto

ifndef INC_CFILES
INC_CFILES	= $(LIB_CFILES)
endif # INC_CFILES

name	= $(subst -, _, $(notdir $(CURDIR)))

build-header-file:
	$(COPY) $(TARGET)-local.h $(INC_TARGET); \
	{ \
	  echo "/* header file built by $(PROTO) */"; \
	  echo "#ifndef $(name)_header_included";\
	  echo "#define $(name)_header_included";\
	  cat $(TARGET)-local.h;\
	  $(PROTOIZE) $(INC_CFILES) | \
	  sed 's/struct _iobuf/FILE/g;s/__const/const/g;/_BUFFER_STATE/d;/__inline__/d;' ;\
	  echo "#endif /* $(name)_header_included */";\
	} > $(INC_TARGET).tmp
	$(MOVE) $(INC_TARGET).tmp $(INC_TARGET)

header:	.header $(INC_TARGET)

# .header carrie all dependencies for INC_TARGET:
.header: $(TARGET)-local.h $(DERIVED_HEADERS) $(LIB_CFILES) 
	$(MAKE) $(GMKNODIR) build-header-file
	touch .header

$(INC_TARGET): $(TARGET)-local.h
	$(RM) .header; $(MAKE) $(GMKNODIR) .header

phase2:	header

clean: header-clean

header-clean:
	$(RM) $(INC_TARGET) .header

INSTALL_INC	+=   $(INC_TARGET)

endif # INC_TARGET

ifdef INSTALL_INC

phase2: .build_inc

$(INC.d):; $(MKDIR) $(INC.d)

.build_inc: $(INSTALL_INC) $(INC.d)
	for f in $(INSTALL_INC) ; do \
	  cmp $$f $(INC.d)/$$f || \
	    $(INSTALL) -m 644 $$f $(INC.d) ; \
	done
	touch $@

clean: inc-clean

inc-clean:
	$(RM) .build_inc

endif # INSTALL_INC

####################################################################### LIBRARY

# ARCH subdirectory
$(ARCH):; test -d $(ARCH) || $(MKDIR) $(ARCH)

clean: arch-clean

arch-clean:
	-$(RM) $(ARCH)/*.o $(ARCH)/lib*.a
	-$(RMDIR) $(ARCH)

ifdef LIB_CFILES
ifndef LIB_OBJECTS
LIB_OBJECTS = $(addprefix $(ARCH)/,$(LIB_CFILES:.c=.o))
endif # LIB_OBJECTS
endif # LIB_CFILES

ifdef LIB_TARGET

$(ARCH)/$(LIB_TARGET): $(LIB_OBJECTS)
	$(ARCHIVE) $(ARCH)/$(LIB_TARGET) $(LIB_OBJECTS)
	ranlib $@

INSTALL_LIB	+=   $(addprefix $(ARCH)/,$(LIB_TARGET))

endif # LIB_TARGET


ifdef INSTALL_LIB

phase2: $(ARCH)

phase3:	.build_lib.$(ARCH)

$(INSTALL_LIB): $(ARCH)

$(LIB.d):
	$(MKDIR) $@

$(LIB.d)/$(ARCH): $(LIB.d)
	$(MKDIR) $@

.build_lib.$(ARCH): $(INSTALL_LIB) $(LIB.d)/$(ARCH)
	for l in $(INSTALL_LIB) ; do \
	  cmp $$l $(LIB.d)/$$l || \
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

install: recompile

# empty dependencies to please compile targets
phase0:
phase1:
phase2:
phase3:
phase4:
phase5:

# binaries
ifdef BIN_TARGET
INSTALL_BIN	+=   $(addprefix $(ARCH)/,$(BIN_TARGET))
endif # BIN_TARGET

ifdef INSTALL_BIN

phase3: .build_bin.$(ARCH)

$(INSTALL_BIN): $(ARCH)

$(BIN.d):
	$(MKDIR) $@

$(BIN.d)/$(ARCH): $(BIN.d)
	$(MKDIR) $@

.build_bin.$(ARCH): $(INSTALL_BIN) $(BIN.d)/$(ARCH)
	$(INSTALL) -m 755 $(INSTALL_BIN) $(BIN.d)/$(ARCH)
	touch $@

clean: bin-clean

bin-clean:
	$(RM) .build_bin.*

endif # INSTALL_BIN

# documentation
ifdef INSTALL_DOC

phase4: .build_doc

$(DOC.d):; $(MKDIR) $(DOC.d)

.build_doc: $(INSTALL_DOC) $(DOC.d)
	$(INSTALL) -m 644 $(INSTALL_DOC) $(DOC.d)
	touch $@

clean: doc-clean

doc-clean:
	$(RM) .build_doc

endif # INSTALL_DOC

# shared
ifdef INSTALL_SHR

phase2: .build_shr 

$(SHR.d):; $(MKDIR) $(SHR.d)

.build_shr: $(INSTALL_SHR) $(SHR.d)
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

.build_utl: $(INSTALL_UTL) $(UTL.d)
	$(INSTALL) -m 755 $(INSTALL_UTL) $(UTL.d)
	touch $@

clean: utl-clean

utl-clean:
	$(RM) .build_utl

endif # INSTALL_UTL
