# 
# $Id$
# 

CPPFLAGS+=	$(PIPS_X11_ADDED_CPPFLAGS)
LDFLAGS+=	$(PIPS_X11_ADDED_LDFLAGS)
X11LIB+=	$(PIPS_X11_ADDED_LIBS)

LIB=		$(ARCH)/libxpomp.a
BIN=		$(ARCH)/xpomp

LOCAL_HEADERS=	gr.h rasterfile.h 
EXPORT_HEADERS=	xpomp_graphic.h xpomp_graphic_F.h
CFILES=		xpomp.c 
M4CFILES=	cgraphic.m4c
DEMO=		test_xpomp.c fractal.f wave.f wave_parameters.h
HPFC=		xpomp_stubs.f

SOURCES=	$(LOCAL_HEADERS) \
		$(EXPORT_HEADERS) \
		$(CFILES) \
		$(M4CFILES) \
		$(DEMO) \
		$(HPFC)

OFILES=		$(ARCH)/cgraphic.o

#
# installation

INSTALL_RTM_DIR:=$(INSTALL_RTM_DIR)/xpomp
INSTALL_BIN_DIR:=$(INSTALL_RTM_DIR)/$(ARCH)

INSTALL_BIN=	$(BIN) $(LIB)
INSTALL_RTM=	$(EXPORT_HEADERS)
INSTALL_SHR=	$(HPFC) xpomp_graphic_F.h xpomp_stubs.direct

# 
# compilation and so

DRUN =	$(LIB) $(BIN) $(ARCH)/fractal $(ARCH)/test_xpomp $(ARCH)/wave

all: run xpomp_stubs.direct
run: $(DRUN)

# cproto:; $(PROTOIZE) xpomp.c

# the direct version of the stubs need not be filtered by hpfc_directives.
xpomp_stubs.direct: xpomp_stubs.f
	# building $@ from $<
	sed 's,^!fcd\$$ fake,      call hpfc9,;\
	     s,^!fcd\$$ io,      call hpfc6,' $< > $@

$(ARCH)/xpomp: $(ARCH)/xpomp.o gr.h
	$(LINK) $@ $(ARCH)/xpomp.o $(X11LIB)

# Deal with different Fortran to C interface call conventions (for strings)

$(PVM_ARCH).m4:
	cp $(PVM_ROOT)/conf/$(PVM_ARCH).m4 $@

M4OPT=	$(PVM_ARCH).m4
cgraphic.c:	$(PVM_ARCH).m4

$(LIB):	$(OFILES) cgraphic.c gr.h
	$(AR) $(ARFLAGS) $(LIB) $(OFILES)
	ranlib $(LIB)

$(ARCH)/test_xpomp : $(ARCH)/test_xpomp.o $(LIB) 
	$(LINK) $@ $+ -lm $(LIB)

$(ARCH)/fractal : $(ARCH)/fractal.o $(LIB) 
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $+ -lm $(LIB)

$(ARCH)/wave : $(ARCH)/wave.o $(LIB) 
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $+ -lm $(LIB)

clean: local-clean
local-clean:
	$(RM) cgraphic.c $(ARCH)/*.o $(DRUN) \
		xpomp_manual.dvi xpomp_stubs.direct 

# that is all
#
