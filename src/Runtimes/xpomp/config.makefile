# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/30 16:02:09 $ 

CPPFLAGS+=	$(PIPS_X11_ADDED_CPPFLAGS)
LDFLAGS+=	$(PIPS_X11_ADDED_LDFLAGS)

LOCAL_LIB=	$(ARCH)/libxpomp.a
CFILES=		cgraphic.c xpomp.c

SOURCES=	$(CFILES)

OFILES=	cgraphic.o
# OFILES:= $(addprefix $(ARCH)/, $(CFILES))

#
# installation

INSTALL_RTM_DIR:=$(INSTALL_RTM_DIR)/xpomp
INSTALL_BIN_DIR:=$(INSTALL_RTM_DIR)/$(ARCH)

INSTALL_BIN=	$(ARCH)/xpomp $(ARCH)/libxpomp.a
INSTALL_RTM=	

# 
# compilation and so.

.SUFFIXES: .c .o

cproto :
	$(PROTOIZE) xpomp.c

all: $(LOCAL_LIB) xpomp test_xpomp fractal


test_xpomp : test_xpomp.o cgraphic.o
	$(CC) -g -o test_xpomp test_xpomp.o cgraphic.o -lm

fractal : fractal.f cgraphic.o
	f77 -g -o fractal fractal.f cgraphic.o -lm

$(LOCAL_LIB):	$(OFILES)
	$(AR) $(ARFLAGS) $(LOCAL_LIB) $(OFILES)
	ranlib $(LOCAL_LIB)


# that is all
#
