.SUFFIXES:      .c .cc .h .l .y .d .tag .mk .exe .in .tested .out


##############################################################
## Standard variable definitions
##############################################################

OBJS= $(SRCS:%.c=$(OBJDIR)/%.o) 

POLY_EXEC= $(SRCS:%.c=%.exe) 

OBJDIR = $(ROOT)/$(OBJ_DIR)
#EXECDIR = $(ROOT)/bin/$(OBJ_DIR)
EXECDIR = $(ROOT)/bin

############################################################## 
## Standard makefile variable for creating platform dependent object
## files
##############################################################

VPATH = $(OBJDIR)

##############################################################
## Rules to compile and to construct executables
##############################################################

# The default compile rule
$(OBJDIR)/%.o : %.c
	@if [ ! -d $(OBJDIR) ]; \
		then mkdir -p $(OBJDIR) ; \
	fi
	$(CC) -c $(CFLAGS) $(EXTRA_FLAGS) $(DEFINES) $(EXTRA_DEFINES) $< -o $@

# The default compile rule to create executable application
.o.exe:
	@if [ ! -d $(EXECDIR) ]; \
		then mkdir -p $(EXECDIR) ; \
	fi
	$(LD) $(LDFLAGS) -o $(EXECDIR)/$*$(BITS).exe $<  \
		$(ROOT)/lib/$(STATIC_LIB) $(EXTRA_LIBS) 

# Recursively, construct all packages
all: execs

# In each package construct the library components
library: $(LIBRARY)

# standard rule to construct a library of object files
$(LIBRARY): $(OBJS)
	ar cr $(OBJDIR)/$(PSTATIC) $(OBJS)
	ranlib $(OBJDIR)/$(PSTATIC)

$(ROOT)/lib/$(STATIC_LIB):
	$(RM) $@
	$(LN_S) $(OBJDIR)/$(PSTATIC) $@

########################################################################
## Recursive rules
########################################################################

# Compile subpackages
package: $(ROOT)/vars.mk
	@if [ "x$(SUBDIRS)" != "x" ]; then \
		set $(SUBDIRS) ; \
		for x do \
		    if [ -r $$x ] ; then \
			( cd $$x ; \
			$(MAKE) $(MFLAGS) $(MAKEVARS) package ;\
			) \
		    fi ; \
		done ; \
	fi
	make library

# Compiles the applications to executables
execs: package
	@if [ "x$(APPDIRS)" != "x" ]; then \
		set $(APPDIRS) ; \
		for x do \
		    if [ -r $$x ] ; then \
			( cd $$x ; \
			$(MAKE) $(MFLAGS) $(MAKEVARS) execs ;\
			) \
		    fi ; \
		done ; \
	fi ;\
	make exe

# Run the tests for the various applications
tests: 
	@if [ "x$(TESTDIRS)" != "x" ]; then \
		set $(TESTDIRS) ; \
		for x do \
		    if [ -r $$x ] ; then \
			( cd $$x ; \
			$(MAKE) $(MFLAGS) $(MAKEVARS) tests ;\
			) \
		    fi ; \
		done ; \
	fi
	make test

# To construct an executable, compile the 'c' file and link with the
# library
exe: $(ROOT)/lib/$(STATIC_LIB) $(POLY_EXEC)

# Remove classes from this directory then subdirectories
clean:
	rm -f $(OBJDIR)/*.o $(OBJDIR)/*.a* $(EXECDIR)/*.exe patch.exe.core 
	rm -f #*# *.*~ *~ *.html *.css 
	rm -f tmpfile junk* *.o TAGS
	@if [ "x$(SUBDIRS)" != "x" ]; then \
		set $(SUBDIRS) ; \
		for x do \
		    if [ -r $$x ] ; then \
			( cd $$x ; \
			$(MAKE) $(MFLAGS) $(MAKEVARS) clean ;\
			) \
		    fi ; \
		done ; \
	fi

# Make the 'javadoc' style documentation
document:
	$(DOXYGEN)/bin/doxygen polylib.doxygen

###########################################################
## Install/UnInstall rules
###########################################################
# main target
install:: install-static install-include install-man install-docs install-exec

# not functional...
#install-shared:
#	$(mkinstalldirs) $(LIBDIR)
#	$(INSTALL) $(OBJ_DIR)/$(PSHARED) $(LIBDIR)/
#	$(RM) $(LIBDIR)/libpolylib$(BITS).$(SHEXT)
#	$(LN_S) $(LIBDIR)/$(PSHARED) $(LIBDIR)/libpolylib$(BITS).$(SHEXT)
#	- $(LDCONFIG)

install-static:
	$(mkinstalldirs) $(LIBDIR)
	$(INSTALL) $(OBJ_DIR)/$(PSTATIC) $(LIBDIR)/
	$(RM) $(LIBDIR)/libpolylib$(BITS).a
	$(LN_S) $(LIBDIR)/$(PSTATIC) $(LIBDIR)/libpolylib$(BITS).a

install-include:
	if [ ! -d "$(INCLUDEDIR)/polylib" ]; then \
		echo "Creating '$(INCLUDEDIR)/polylib' directory"; \
		$(mkinstalldirs) $(INCLUDEDIR)/polylib ;\
	$(INSTALL_DATA) ./include/polylib/*.h $(INCLUDEDIR)/polylib ;\
	fi

install-man:
# to be done...

install-docs:
	$(mkinstalldirs) $(DOCSDIR)
	$(INSTALL_DATA) doc/*.gz $(DOCSDIR)/
	$(mkinstalldirs) $(DOCSDIR)/examples/ehrhart
	$(INSTALL_DATA) Test/ehrhart/*.in $(DOCSDIR)/examples/ehrhart

install-exec: $(POLY_EXEC)
	$(mkinstalldirs) $(BINDIR)
	$(INSTALL) $(ROOT)/bin/$(OBJ_DIR)/*.exe $(BINDIR)

# Get the Platform dependent VARIABLES
include $(ROOT)/vars.mk
