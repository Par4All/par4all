# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/09/03 23:25:45 $ 

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

OFILES=		cgraphic.o

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

all: run doc
run: $(LIB) $(BIN) fractal test_xpomp wave
doc: $(INSTALL_DOC) xpomp_manual.html

# cproto:; $(PROTOIZE) xpomp.c

xpomp: $(ARCH)/xpomp.o
	$(LINK) $@ $+ $(X11LIB)

# Deal with different Fortran to C interface call conventions (for strings)
M4OPT=$(PVM_ROOT)/conf/$(PVM_ARCH).m4

$(LIB):	$(OFILES) cgraphic.c
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
	$(RM) cgraphic.c $(ARCH)/*.o $(BIN) $(LIB) $(ARCH)/fractal $(ARCH)/test_xpomp
	$(RM) -r xpomp_manual xpomp_manual.html xpomp_manual.ps \
		xpomp_manual.dvi

# that is all
#
