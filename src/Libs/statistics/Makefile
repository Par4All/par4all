#	---------------------------------------------------------------
#	---------------------------------------------------------------
#
#				    WARNING
#
#		  This makefile has been automatically generated
#
#			       Do not modify it
#
#        You can modify the file config.makefile in the same directory
#
#	---------------------------------------------------------------
#	---------------------------------------------------------------
#
# Copyright (C) Ecole des Mines De Paris
#               Centre de Recherche en Informatique   
#
# This file is part of PIPS
#
# : pips-makemake,v 1.335 1999/03/30 09:19:55 coelho Exp $
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY.  No author or distributor accepts responsibility to anyone for
# the consequences of using it or for whether it serves any particular
# purpose or works at all, unless he/she says so in writing.
#
# DEFAULTS, may be overwritten by the config.makefile content.
#
# the root file of the soft which is being compiled
ROOT= $(PIPS_ROOT)
#    
TARGET=		statistics
TARGET_DIR=	$(ROOT)/Src/Libs/statistics
#
# default architecture is taken from PIPS_ARCH then from PIPS_ARCH 
# environment variable, else the default is "." (dot). 
ifdef PIPS_ARCH
ARCH =	$(PIPS_ARCH)
else
ifdef PIPS_ARCH
ARCH =	$(PIPS_ARCH)
else
ARCH =	.
endif
endif
#
# installation directories for quick-install
#
INSTALL_INC_DIR= $(ROOT)/Include
INSTALL_SHR_DIR= $(ROOT)/Share
INSTALL_DOC_DIR= $(ROOT)/Doc
INSTALL_HTM_DIR= $(ROOT)/Html
INSTALL_UTL_DIR= $(ROOT)/Utils
INSTALL_RTM_DIR= $(ROOT)/Runtime
INSTALL_LIB_DIR= $(ROOT)/Lib/$(ARCH)
INSTALL_BIN_DIR= $(ROOT)/Bin/$(ARCH)
#
CONFIG	= config.makefile
MAKEFILE= Makefile
# the umask within pips is fixed here!
# however I cannot fix the group at this level.
UMASK   = 002
SHELL	= umask $(UMASK) ; exec /bin/sh
COPY	= cp
MOVE	= mv -f
GMKNODIR= --no-print-directory
INSTALL = pips_install_file
INSTSRC = install_pips_sources
TMPFILE = .tmp
DEPFILE = $(TMPFILE)
#
# default targets...
all: generate_makefile_if_necessary_and_possible
#
recompile: depend header all
recompile: generate_makefile_if_necessary_and_possible
#
full-clean: clean clean-all-compiled
clean-all-compiled: 
	-for d in `echo $(ROOT)/Include/makefile_macros.* | \
		sed 's,[-_a-zA-z0-9/]*/makefile_macros\.,,g'`; do \
	  test -d $$d && { \
	    echo "cleaning $$d"; \
	    $(RM) $$d/*.[oa] $$d/*pips Makefile.depend.$$d; \
	    test "`ls -A $$d`" || rmdir $$d; } ; \
	done
#
OLD:; mkdir OLD
RCS:; mkdir RCS
# .PHONY is a gnu extension...
.PHONY: install all depend header recompile clobber count info delta
.PHONY: generate_makefile_if_necessary_and_possible
# to override SUN/GNU default rule
.SCCS_GET:
# %:SCCS/s.%:
.SUFFIXED: .tex .dvi .ps .c .h .o .w .a .html .ftex .newgen .m4h .m4c .m4f
# .INIT is a SUN extension...
.INIT: 
check_install:
# is this appropriate?
# .DEFAULT:; @echo "Makefile [$(FWD_ROOT)] *** no rule for target !" >&2
#
# Directory redirection to $(PIPS_DIRECTORY_FOR_$(ARCH)_BINARIES)
# if defined: a link is created there... enabling to store binaries
# somewhere else even if it looks like a local subdirectory.
# else the default is simply to make a subdir.
#
BIN_DIRECTORIES=$(PIPS_DIRECTORY_FOR_$(ARCH)_BINARIES)
ifeq ($(BIN_DIRECTORIES),)
# nowhere to link
$(ARCH):; test -d $(ARCH) || mkdir $(ARCH)
else
# somewhere to link!
$(ARCH):
	# creating remote $(ARCH) directory... (arch added for relative links)
	rdir=$(BIN_DIRECTORIES)/Deve/PIPS/Libs/statistics/$(ARCH);\
	test -d $$rdir || mkdir -p $$rdir ; \
	ln -s $$rdir $(ARCH);
endif
#
include $(ROOT)/Include/define_libraries.make
include $(ROOT)/Include/makefile_macros.$(ARCH)
include $(CONFIG)
#
# if some C files, there is a executable target to build.
all: $(ARCH) $(ARCH)/lib$(TARGET).a
#
# The software architecture (compilers and options...) used.
UTC_DATE = $(shell date -u | tr ' ' '_')
CPPFLAGS += -DSOFT_ARCH='"$(ARCH)"'
#
# {C,CPP,LD,L,Y}OPT macros allow to *add* things from the command line
# as gmake CPPOPT="-DFOO=bar" ... that will be added to the defaults
# a typical interesting example is to put -pg in {C,LD}OPT
#
PREPROC=  $(CC) -E $(CANSI) $(CPPOPT) $(CPPFLAGS)
COMPILE=  $(CC) $(CANSI) $(CFLAGS) $(COPT) $(CPPOPT) $(CPPFLAGS) -c
F77COMPILE=$(FC) $(FFLAGS) $(FOPT) -c
LINK=	  $(LD) $(LDFLAGS) $(LDOPT) -o
SCAN=	  $(LEX) $(LFLAGS) $(LOPT) -t
TYPECHECK=$(LINT) $(LINTFLAGS) $(CPPFLAGS) $(LINT_LIBS)
PARSE=	  $(YACC) $(YFLAGS) $(YOPT)
ARCHIVE=  $(AR) $(ARFLAGS)
PROTOIZE= $(PROTO) $(PRFLAGS) -E "$(PREPROC) -DCPROTO_IS_PROTOTYPING"
M4FILTER= $(M4) $(M4OPT) $(M4FLAGS)
MAKEDEP=  $(CC) $(CMKDEP) $(CANSI) $(CFLAGS) $(COPT) $(CPPOPT) $(CPPFLAGS)
#
# for easy debugging... e.g. gmake ECHO='something' echo
echo:; @echo $(ECHO)
#
# Default rules
ifneq ($(ARCH),.)
%: $(ARCH)/%
%.o: $(ARCH)/%.o
%.a: $(ARCH)/%.a
endif
# Java:
%.class: %.java; javac $<
%.h: %.class; javah -jni $*
# C:
$(ARCH)/%.o: %.c ; $(COMPILE) $< -o $@
# Fortran 77:
$(ARCH)/%.o: %.f ; $(F77COMPILE) $< -o $@

# if LATEX, MAKEIDX, LX2HTML, DVIPS RMAN and so are not avaible, 
# the following rules would not prevent compilation...
%.tex: %.ftex
	# building $@ from $<
	sed -f $(PIPS_ROOT)/Utils/accent.sed $< > $@
#
%.dvi: %.tex
	# building $@ from $<
	-grep '\\makeindex' $*.tex && touch $*.ind
	$(LATEX) $<
	-grep '\\bibdata{' $*.aux && { $(BIBTEX) $* ; $(LATEX) $< ;}
	test ! -f $*.idx || { $(MAKEIDX) $*.idx ; $(LATEX) $< ;}
	$(LATEX) $<
	touch $@
#
%.ps: %.dvi
	# building $@ from $<
	$(DVIPS) -o $@ $<
	touch $@
#
%.html: %.l
	# building $@ from $<
	$(NROFF) -man $< | $(RMAN) -f html > $@
	touch $@
#
%.ps: %.l
	# building $@ from $<
	$(GROFF) -tpe -man $< > $@
#
# Add the PIPS icon in the background of every page:
#?cd $*; for f in *.html ; do \
# sed 's,^<BODY,<BODY background="http://www.cri.ensmp.fr/pips/images/logo-pips-embossed-bright-small.jpg" ,' 2928f > 2928f.bg; \
# mv 2928f.bg 2928f; done
# the .html file is not a real one, it is just used as a dependency.
%.html: %.tex
	# building $@ from $<
	$(RM) -r $* $@ ; 	mkdir $* ;	$(LX2HTML) $(L2HFLAGS) $< ;	# Does not removed the .pl files since it is used
	# for cross-referencing between HTML articles:
	# Does not removed the .css files since it is included 
	# in the generated HTML articles:
	$(RM) $*/*.aux $*/*.idx $*/*.log $*/*.tex 	    $*/.[A-Z]* ;
	# local .ps.gz if available
	test ! -f $*.ps || { cp $*.ps $* ; $(ZIP) $*/$*.ps ; }
	# index.html
	{ cd $* ; 	  ln -s $@ index.html ; 	  chmod -R a+r-wx * ; 	  chmod a+rx,ug+w,o-w . ; } 
	echo "<html><a href=$*/index.html>index</a></html>" > $@
#
%.newgen: %.tex
	# building $@ from $<
	$(RM) $@
	remove-latex-comments $<
	chmod a-w $@
#
%.o: %.f
	$(F77COMPILE) $< -o $@
#
%.f: %.m4f
	# building $@ from $<
	$(M4FILTER) $(M4FOPT) $< > $@
#
%.c: %.m4c
	# building $@ from $<
	$(M4FILTER) $(M4COPT) $< > $@
#
%.h: %.m4h
	# building $@ from $<
	$(M4FILTER) $(M4HOPT) $< > $@
# MISC
.runable: $(SCRIPTS)
	# protected because only the owner can chmod files
	-[ ! "$(SCRIPTS)" ] || chmod a+rx $(SCRIPTS)
	touch .runable
#
quick-install: .runable
#
#
# prefixed objects:
LIB_OBJECTS:=$(addprefix $(ARCH)/, $(LIB_OBJECTS))
#
# if some C files...
LIB_TARGET=	$(ARCH)/lib$(TARGET).a
ifneq ($(ARCH),.)
lib$(TARGET).a: $(LIB_TARGET)
endif
BIN_TARGET=	$(ARCH)/$(TARGET)
INC_TARGET=	$(TARGET).h
#
INSTALL_INC+=	$(INC_TARGET)
INSTALL_LIB+=	$(LIB_TARGET)
INSTALL_BIN+=	
#

.PRECIOUS: $(LIB_TARGET) $(BIN_TARGET)

$(LIB_TARGET): $(INC_TARGET) $(DERIVED_CFILES) $(DERIVED_HEADERS)

$(LIB_TARGET): $(LIB_OBJECTS) $(TARGET_OBJECTS)
	$(RM) $(LIB_TARGET)
	$(ARCHIVE) $(LIB_TARGET) $(LIB_OBJECTS) $(TARGET_OBJECTS)
	-$(RANLIB) $(LIB_TARGET)

check_install:
	@if test `echo *.a | wc -w` -gt 1 ; \
	then \
	  echo "install failed: additional local libraries prevented it" >&2;\
	  exit 1;\
	fi


CFILES=		$(LIB_CFILES)
OBJECTS=	$(LIB_OBJECTS)
INCLUDES=	$(LIB_HEADERS)
SOURCES+= 	$(INCLUDES) $(CFILES) $(DOCS) $(LIB_MAIN)
#

clean-compiled:
	$(RM) $(OBJECTS) $(LIB_TARGET) $(BIN_TARGET) \
		$(ARCH)/pips $(ARCH)/[ftw]pips $(LIB_TARGET:.a=.so)

clean: generate_makefile_if_necessary_and_possible clean-compiled
	-$(RM) TAGS $(BIN_TARGET) $(DERIVED_HEADERS) $(DERIVED_CFILES) \
	       $(DERIVED_FILES) *.~[0-9]~ $(LIB_IFILES) $(INC_TARGET) core

INC_CFILES += $(LIB_CFILES)

#

test: all
	#
	# building a local PIPS
	#
	$(RM) $(ARCH)/pips
	$(LINK) $(ARCH)/pips $(ROOT)/Lib/$(ARCH)/$(PIPS_MAIN) \
	    -lpips $(PIPS_LIBS) 
	@echo 

ttest: all
	#
	# building a local TPIPS
	#
	$(RM) $(ARCH)/tpips
	$(LINK) $(ARCH)/tpips $(ROOT)/Lib/$(ARCH)/$(TPIPS_MAIN) \
	    -ltpips $(PIPS_LIBS) $(TPIPS_ADDED_LIBS)
	@echo 

wtest: all
	#
	# building a local WPIPS
	#
	$(RM) $(ARCH)/wpips
	$(LINK) $(ARCH)/wpips $(WPIPS_ADDED_LDFLAGS) \
	    $(ROOT)/Lib/$(ARCH)/$(WPIPS_MAIN) \
	    -lwpips $(PIPS_LIBS) $(WPIPS_ADDED_LIBS)
	@echo 

ftest: all
	#
	# building a local Full PIPS
	#
	$(RM) $(ARCH)/fpips
	$(LINK) $(ARCH)/fpips $(FPIPS_ADDED_LDFLAGS) \
	    $(ROOT)/Lib/$(ARCH)/$(FPIPS_MAIN) \
	    -lfpips $(FPIPS_ADDED_LIBS) \
	    $(PIPS_LIBS) 
	#
	# creating links
	#
	$(RM) $(ARCH)/pips $(ARCH)/[tw]pips
	ln $(ARCH)/fpips $(ARCH)/pips 
	ln $(ARCH)/fpips $(ARCH)/tpips 
	ln $(ARCH)/fpips $(ARCH)/wpips 
	#
	@echo 

# obsolete rule I guess. test is better. FC.
$(BIN_TARGET): $(ARCH)/$(LIB_MAIN:.c=.o) $(LIB_TARGET)
	#
	# building $@
	#
	$(LINK) $(BIN_TARGET) $< $(PIPS_LIBS)
	@echo 

clobber: clean ; -Delete


ALL_SOURCES =	$(SOURCES) $(CONFIG)

$(TARGET_DIR):;	mkdir $(TARGET_DIR)

.quick-install: all
	#
	# installing files 
	# 
	# plain files
	@test ! "$(INSTALL_INC)" || \
	    $(INSTALL) -f $(INSTALL_INC_DIR) $(INSTALL_INC) 
	@test ! "$(INSTALL_DOC)" || \
	    $(INSTALL) -f $(INSTALL_DOC_DIR) $(INSTALL_DOC) 
	@test ! "$(INSTALL_HTM)" || \
	    $(INSTALL) -f $(INSTALL_HTM_DIR) $(INSTALL_HTM) 
	@test ! "$(INSTALL_UTL)" || \
	    $(INSTALL) -f $(INSTALL_UTL_DIR) $(INSTALL_UTL) 
	@test ! "$(INSTALL_SHR)" || \
	    $(INSTALL) -f $(INSTALL_SHR_DIR) $(INSTALL_SHR)
	@test ! "$(INSTALL_RTM)" || \
	    $(INSTALL) -f $(INSTALL_RTM_DIR) $(INSTALL_RTM)
	# libs...
	@test ! "$(INSTALL_LIB)" || \
	    $(INSTALL) -r $(RANLIB) $(INSTALL_LIB_DIR) $(INSTALL_LIB)
	# executables...
	@test ! "$(INSTALL_BIN)" || \
	    $(INSTALL) -x $(INSTALL_BIN_DIR) $(INSTALL_BIN)
	touch .quick-install

tar:
	# taring $(TARGET)
	cd ..; \
	    $(TAR) cf $(TARGET).tar \
		$(addprefix $(TARGET)/,$(ALL_SOURCES) Makefile) ; \
	    $(ZIP) $(TARGET).tar

quick-install: .quick-install

.makefile: $(ROOT)/Utils/pips-makemake
	# 
	# regenerating Makefile
	#
	pips-makemake  -l; touch .makefile

generate_makefile_if_necessary_and_possible:
	@[ ! -x $(ROOT)/Utils/pips-makemake ] || \
	    $(MAKE) $(GMKNODIR) .makefile 

count:;	@wc -l $(ALL_SOURCES)

#
# some quick entries for SCCS:
info:
	-test -d SCCS && sccs info
	-test -d RCS && rlog -L -l RCS/*

# try to delta all checked out files
delta:
	-test -d SCCS && test ! "`sccs tell`" || sccs delget `sccs tell`
	-test -d RCS && ci -u `rlog -L -R RCS/*`
check_sccs:
	@if [ -d SCCS ] ; then \
	    [ ! "`sccs tell`" ] || \
	    { echo "SCCS files locked, cannot install"; exit 1;} fi ;

check_rcs:
	@if [ -d RCS ] ; then \
	    [ ! "`rlog -L -R RCS/*`" ] || \
	    { echo "RCS files locked, cannot install"; exit 1;} fi ;

# install installs sources, so depends on sources. FC
# to insure the library is compiled before hand, a make is added.
.install: $(ALL_SOURCES) 
	#
	# first check
	#
	$(MAKE) check_install check_sccs check_rcs
	#
	# make target directory if necessary
	#
	$(MAKE) $(GMKNODIR) all $(TARGET_DIR) 
	#
	# installing new source files 
	#
	@echo ""; echo; echo -n "Description of changes: " ; \
	read x ; \
	$(INSTSRC) -rv -s `pwd` -t $(TARGET_DIR) -y "$$x" $(ALL_SOURCES)
	-@cd $(TARGET_DIR) ; pips-makemake  -l -S
	touch .install
	$(MAKE) quick-install

install:  .install

# diff between dev and prod sources:
diff:
	#
	# showing diff between the local and installed versions
	#
	-@for f in $(ALL_SOURCES) ; do \
	  echo "DIFFERENCES BETWEEN DEV AND PROD FOR $$f:" ; \
	  $(DIFF) $$f $(TARGET_DIR)/$$f ; \
	done

#

$(ROOT)/Lib:; mkdir $@

lint:; $(TYPECHECK) $(CFILES)|sed '/possible pointer alignm/d;/gen_alloc/d'

TAGS: $(INCLUDES) $(CFILES)
	etags $(INCLUDES) $(CFILES)
	ctags $(INCLUDES) $(CFILES)

$(TARGET)-local.h:
	[ -f $(TARGET)-local.h ] || { \
	    test -f RCS/$(TARGET)-local.h,v && co $(TARGET)-local.h ; \
	    touch $(TARGET)-local.h ; }

# headers are not generated automatically, bacause it should be
# done each time a c file is touched, which is too much...
# thus it does depend only of the -local.h.
# header file reconstruction should be required explicitely
# with the header rule.
# some filtering is performed to avoid variations that depends on
# includes...

build-header-file:
	# 
	# building the header file...
	#
	$(COPY) $(TARGET)-local.h $(INC_TARGET);\
	{ \
	echo "/* header file built by $(PROTO) */";\
	echo "#ifndef statistics_header_included";\
	echo "#define statistics_header_included";\
	cat $(TARGET)-local.h;\
	$(PROTOIZE) $(INC_CFILES) | \
	  sed 's/struct _iobuf/FILE/g;s/__const/const/g;/_BUFFER_STATE/d;/__inline__/d;' ;\
	echo "#endif /* statistics_header_included */";\
	} > $(INC_TARGET).tmp
	$(MOVE) $(INC_TARGET).tmp $(INC_TARGET)

# .header carrie all dependencies for INC_TARGET:
.header: $(TARGET)-local.h $(DERIVED_HEADERS) $(INC_CFILES) 
	$(MAKE) $(GMKNODIR) build-header-file ; touch .header

# checks for the header file, and build if necessary...

# header: checks for INC_TARGET dependencies (.header) and for
# INC_TARGET existance !
header: .header $(INC_TARGET)

$(INC_TARGET): $(TARGET)-local.h 
	$(RM) .header; $(MAKE) $(GMKNODIR) .header

ifneq ($(ARCH),.)
lib$(TARGET).a: $(ARCH) $(ARCH)/lib$(TARGET).a
endif

#

# indirect depend bacause I do not trust make for the inclusion of
# Makefile.depend if that very file does not exist.

.depend: $(CFILES) $(DERIVED_CFILES) Makefile.depend.$(ARCH)
	touch $@

depend: .depend

# default rule if the dependence file is not there...
# forces the regeneration:
# - the dependences' generation needs the headers
# - the headers are generated thru the Makefile.
# - the Makefile needs the dependences.
# how to get out of this?
Makefile.depend.$(ARCH):
	#
	# generating the dependences for $(ARCH)
	#
	# creates an empty dependence file, it is ok for going on and
	# will prevent any recursion for generating the same dependences.
	touch $@ 
	# the headers are needed. 
	test ! "$(INC_TARGET)" -o -f "$(INC_TARGET)" || \
	    $(MAKE) $(GMKNODIR) $(INC_TARGET)
	# may need the derived files
	test ! "$(DERIVED_CFILES)" || $(MAKE) $(GMKNODIR) $(DERIVED_CFILES)
	# now it's okay fo building all the dependences.
	# well, the how depends on the compiler...:-(
	test ! "$(CFILES)$(DERIVED_CFILES)" || \
	{ $(MAKEDEP) $(CFILES) $(DERIVED_CFILES) > $(TMPFILE) ; \
	  sed 's,^\(.*\.o:\),$$(ARCH)/\1,' $(DEPFILE) > $@ ; \
	  $(RM) $(TMPFILE) $(DEPFILE) ; }

#
# dependences are stored in another file which is simply included.
# the file differs for each "architectures"
include Makefile.depend.$(ARCH)
