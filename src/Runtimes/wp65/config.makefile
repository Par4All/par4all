# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1995/09/19 22:28:19 $ 
#

# How desperate I am to put that:-)
# Otherwise the generated library will be in trouble for linking 
# with SUN's f77... I would need to append the library path an other
# related gcc linraries...
CC=acc
CFLAGS=-O -g

LOCAL_LIB=	libwp65runtime.a
CFILES=		lance_wp65.c

SOURCES=	$(CONFIG_FILE) \
		$(CFILES) \
		compile_wp65 \
		Makefile.compile_wp65

OFILES=	$(CFILES:.c=.o)

#
# installation

INSTALL_EXE_DIR=	$(PIPS_UTILDIR)

INSTALL_FILE=	Makefile.compile_wp65
INSTALL_LIB=	$(LOCAL_LIB)
INSTALL_EXE=	compile_wp65

#
# pvm headers:

CPPFLAGS+=	-I$(PVM_ROOT)/include

# 
# compilation and so.

.SUFFIXES: .c .o

all: $(LOCAL_LIB) .runable

.runable: compile_wp65
	chmod a+x compile_wp65
	touch $@

.c.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

$(LOCAL_LIB):	$(OFILES)
	$(AR) $(ARFLAGS) $(LOCAL_LIB) $(OFILES)
	ranlib $(LOCAL_LIB)

clean:
	-$(RM) *~ *.o

# that is all
#
