# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/30 18:15:57 $ 

ifeq ($(FC),g77)
CPPFLAGS+=	-DCOMPILE_FOR_G77
endif

CPPFLAGS+=	$(PIPS_X11_ADDED_CPPFLAGS)
LDFLAGS+=	$(PIPS_X11_ADDED_LDFLAGS)

LIB=		$(ARCH)/libxpomp.a
BIN=		$(ARCH)/xpomp

LOCAL_HEADERS=	gr.h rasterfile.h 
EXPORT_HEADERS=	xpomp_graphic.h xpomp_graphic_F.h
CFILES=		cgraphic.c xpomp.c 
DEMO=		test_xpomp.c fatal.f 

SOURCES=	$(LOCAL_HEADERS) \
		$(EXPORT_HEADERS) \
		$(CFILES) \
		$(DEMO)

OFILES=		cgraphic.o

#
# installation

INSTALL_RTM_DIR:=$(INSTALL_RTM_DIR)/xpomp
INSTALL_BIN_DIR:=$(INSTALL_RTM_DIR)/$(ARCH)

INSTALL_BIN=	$(BIN) $(LIB)
INSTALL_RTM=	$(EXPORT_HEADERS)

# 
# compilation and so

all: $(LIB) $(ARCH)/xpomp test_xpomp fractal

cproto :
	$(PROTOIZE) xpomp.c

xpomp: $(ARCH)/xpomp.o
	$(LINK) $@ $+ $(PIPS_X11_ADDED_LIBS)

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

# that is all
#
