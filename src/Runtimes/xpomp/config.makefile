# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/31 13:16:18 $ 

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
CFILES=		cgraphic.c xpomp.c 
DEMO=		test_xpomp.c fractal.f 
HPFC=		xpomp_fake.f
DOC=		xpomp_manual.tex xPOMP_window_explained.eps
SOURCES=	$(LOCAL_HEADERS) \
		$(EXPORT_HEADERS) \
		$(CFILES) \
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
INSTALL_HTM=	xpomp_manual.html xpomp_manual

# 
# compilation and so

all: run doc
run: $(LIB) $(BIN) fractal test_xpomp
doc: $(INSTALL_DOC) $(INSTALL_HTM)

# cproto:; $(PROTOIZE) xpomp.c

xpomp: $(ARCH)/xpomp.o
	$(LINK) $@ $+ $(X11LIB)

$(LIB):	$(OFILES)
	$(AR) $(ARFLAGS) $(LIB) $(OFILES)
	ranlib $(LIB)

$(ARCH)/test_xpomp : $(ARCH)/test_xpomp.o $(LIB) 
	$(LINK) $@ $+ -lm $(LIB)

$(ARCH)/fractal : $(ARCH)/fractal.o $(LIB) 
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $+ -lm $(LIB)

clean: local-clean
local-clean:
	$(RM) $(ARCH)/*.o $(BIN) $(LIB) $(ARCH)/fractal $(ARCH)/test_xpomp
	$(RM) -r xpomp_manual xpomp_manual.html xpomp_manual.ps \
		xpomp_manual.dvi

# that is all
#
