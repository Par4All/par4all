################################################
## PolyLib Makefile.  Version : 5.0           ##
## Vincent Loechner, 1998/99/00.              ##
################################################
## Main targets of this makefile are :
## all, libs, install-libs, exec, install[-exec], uninstall, 
## 32ln-exec, 64ln-exec (to link executables without the extra suffix)
## Other targets : 32, 64, typecheck.

POLYLIBDIR=.
include vars.mk


mkinstalldirs = $(SHELL) ./mkinstalldirs


PEXEC = \
	testlib \
	polytest \
	c2p \
	r2p \
	findv \
	pp \
	disjoint_union_sep \
	disjoint_union_adj \
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
	@echo "'make [all]' to build the libs,  and build the executables (default)."
	@echo "'make libs' to build the libraries ($(LIBS_TO_BUILD))."
	@echo "'make install-libs' to install  somewhere (in $(prefix))."
	@echo "'make install' to build and install somewhere (in $(prefix))."
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
	if [ ! -d "$(INCLUDEDIR)/polylib" ]; then \
		echo "Creating '$(INCLUDEDIR)/polylib' directory"; \
		$(mkinstalldirs) $(INCLUDEDIR)/polylib ;\
		$(INSTALL_DATA) ./include/polylib/* $(INCLUDEDIR)/polylib ;\
	fi

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
mlibs: $(LIBSTYPE_TO_BUILD)
# lib-shared lib-static

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
	cp $(ARITH_DIR)/$(OBJ_DIR)/errors.o $(OBJ_DIR)/
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
cvsclean:
	$(RM) -r CVS */CVS */*/CVS

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
$(OBJ_DIR)/disjoint_union_sep$(EXEC_EXTRA_SUFFIX): $(PHEADERS) $(LIB) \
			$(POLYLIB_SRC)/disjoint_union_sep.c
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ \
		$(POLYLIB_SRC)/disjoint_union_sep.c $(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)
$(OBJ_DIR)/disjoint_union_adj$(EXEC_EXTRA_SUFFIX): $(PHEADERS) $(LIB) \
			$(POLYLIB_SRC)/disjoint_union_adj.c
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ \
		$(POLYLIB_SRC)/disjoint_union_adj.c $(EXEC_EXTRA_LIBS) $(EXTRA_LIBS)
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
#vars.mk: vars.mk.in
#	./configure

########################################################################
## END
########################################################################



