#
# $Id$
#
# configuration for FPIPS.
#

LIB_CFILES	= fpips.c
LIB_MAIN	= fpips_main.c
LIB_OBJECTS	= $(LIB_CFILES:.c=.o)

#
# linking fpips.

LDFLAGS  +=	$(WPIPS_ADDED_LDFLAGS) \
		$(PIPS_X11_ADDED_LDFLAGS)

TARGET_LIBS =	-lpips -ltpips -lwpips \
		$(PIPS_LIBS) \
		$(TPIPS_ADDED_LIBS) \
		$(WPIPS_ADDED_LIBS) \
		$(PIPS_X11_ADDED_LIBS)

#
# real lings for the name

local-clean:
	$(RM) $(ARCH)/pips $(ARCH)/tpips $(ARCH)/wpips

clean: local-clean

ln: local-clean
	#
	# links
	#
	ln $(ARCH)/fpips $(ARCH)/pips 
	ln $(ARCH)/fpips $(ARCH)/tpips 
	ln $(ARCH)/fpips $(ARCH)/wpips 

#
#
