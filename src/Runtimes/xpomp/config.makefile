# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/27 22:38:23 $ 

CPPFLAGS+=	$(PIPS_X11_ADDED_CPPFLAGS)
LDFLAGS+=	$(PIPS_X11_ADDED_LDFLAGS)

LOCAL_LIB=	libxhpfcruntime.a
CFILES=		cgraphic.c xhpfc.c

SOURCES=	$(CONFIG_FILE) \
		$(CFILES)

OFILES=	cgraphic.c
# OFILES:= $(addprefix $(ARCH)/, $(CFILES))

#
# installation

INSTALL_RTM_DIR:=$(INSTALL_RTM_DIR)/$(ARCH)
INSTALL_RTM=	$(ARCH)/xhpfc $(ARCH)/libxhpfcruntime.a


# 
# compilation and so.

.SUFFIXES: .c .o

cproto :
	$(PROTOIZE) xhpfc.c

all: $(LOCAL_LIB) xhpfc test_xhpfc fractal


test_xhpfc : test_xhpfc.o cgraphic.o
	$(CC) -g -o test_xhpfc test_xhpfc.o cgraphic.o -lm

fractal : fractal.f cgraphic.o
	f77 -g -o fractal fractal.f cgraphic.o -lm

$(LOCAL_LIB):	$(OFILES)
	$(AR) $(ARFLAGS) $(LOCAL_LIB) $(OFILES)
	ranlib $(LOCAL_LIB)


# that is all
#
