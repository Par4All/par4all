# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/09/06 10:55:56 $ 

# expected from makefile macros
ifeq ($(FC),g77)
CPPFLAGS+=	-DCOMPILE_FOR_G77
endif

# expected from defines...
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
HPFC=		xpomp_fake.f
DOC=		xpomp_manual.tex xPOMP_window_explained.eps

SOURCES=	$(LOCAL_HEADERS) \
		$(EXPORT_HEADERS) \
		$(CFILES) \
		$(M4CFILES) \
		$(DEMO) \
		$(DOC)

OFILES=		$(ARCH)/cgraphic.o

#
# installation

INSTALL_RTM_DIR:=$(INSTALL_RTM_DIR)/xpomp
INSTALL_BIN_DIR:=$(INSTALL_RTM_DIR)/$(ARCH)

INSTALL_BIN=	$(BIN) $(LIB)
INSTALL_RTM=	$(EXPORT_HEADERS)
INSTALL_SHR=	$(HPFC) xpomp_graphic_F.h
INSTALL_DOC=	xpomp_manual.ps 
INSTALL_HTM=	xpomp_manual

# 
# compilation and so

DRUN =	$(LIB) $(BIN) $(ARCH)/fractal $(ARCH)/test_xpomp $(ARCH)/wave
DDOC =	$(INSTALL_DOC) xpomp_manual.html xpomp_manual/fractal.f

all: run doc
run: $(DRUN)
doc: $(DDOC)

xpomp_manual/fractal.f: xpomp_manual.html fractal.f
	cp fractal.f xpomp_manual

# cproto:; $(PROTOIZE) xpomp.c

$(ARCH)/xpomp: $(ARCH)/xpomp.o gr.h
	$(LINK) $@ $(ARCH)/xpomp.o $(X11LIB)

# Deal with different Fortran to C interface call conventions (for strings)
M4OPT=$(PVM_ROOT)/conf/$(PVM_ARCH).m4

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
	$(RM) cgraphic.c $(ARCH)/*.o $(DRUN) xpomp_manual.dvi
	$(RM) -r $(DDOC) 

# that is all
#
