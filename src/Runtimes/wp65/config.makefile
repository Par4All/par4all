# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/18 09:11:28 $ 

CPPFLAGS+=	$(WP65_ADDED_CPPFLAGS)

LOCAL_LIB=	libwp65runtime.a
CFILES=		lance_wp65.c

SOURCES=	$(CONFIG_FILE) \
		$(CFILES) \
		compile_wp65 \
		Makefile.compile_wp65

OFILES=	$(CFILES:.c=.o)
# OFILES:= $(addprefix $(ARCH)/, $(CFILES))

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

$(LOCAL_LIB):	$(OFILES)
	$(AR) $(ARFLAGS) $(LOCAL_LIB) $(OFILES)
	ranlib $(LOCAL_LIB)

clean-compiled: clean
clean:
	-$(RM) *~ *.o $(LOCAL_LIB)

# that is all
#
