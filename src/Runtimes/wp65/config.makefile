# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/08/21 18:19:23 $ 
#

CPPFLAGS+=	-D COMPILE_FOR_G77

SCRIPTS = 	compile_wp65
CFILES=		lance_wp65.c
MFILE=		Makefile.compile_wp65

SOURCES=	$(CFILES) $(SCRIPTS) $(MFILE)

LOCAL_LIB=	$(PVM_ARCH)/libwp65runtime.a

OFILES=	$(addprefix $(PVM_ARCH)/, $(CFILES:.c=.o))

#
# installation

INSTALL_RTM_DIR:= $(INSTALL_RTM_DIR)/wp65
INSTALL_LIB_DIR= $(INSTALL_RTM_DIR)/$(PVM_ARCH)

INSTALL_LIB=	$(LOCAL_LIB)
INSTALL_RTM=	$(MFILE) $(SCRIPTS)
INSTALL_SHR=	$(SCRIPTS)

#
# pvm headers:

CPPFLAGS+=	-I$(PVM_ROOT)/include

# 
# compilation and so.

$(PVM_ARCH)/%.o: %.c
	$(COMPILE) $< -o $@

all: $(PVM_ARCH) $(LOCAL_LIB) .runable

$(PVM_ARCH):; mkdir $@

$(LOCAL_LIB):	$(OFILES)
	$(AR) $(ARFLAGS) $(LOCAL_LIB) $(OFILES)
	$(RANLIB) $(LOCAL_LIB)

clean-compiled: clean
clean: local-clean
local-clean:
	-$(RM) *~ $(OFILES) $(LOCAL_LIB)

# that is all
#
