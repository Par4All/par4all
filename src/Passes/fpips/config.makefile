#
# $Id$
#
# configuration for FPIPS.
#

LIB_CFILES	= fpips.c
LIB_MAIN	= main_fpips.c
LIB_OBJECTS	= $(LIB_CFILES:.c=.o)

#
# linking fpips.

CPPFLAGS +=	$(FPIPS_ADDED_CPPFLAGS) -DUTC_DATE='"$(UTC_DATE)"'

LDFLAGS  +=	$(FPIPS_ADDED_LDFLAGS)

TARGET_LIBS =	$(FPIPS_ADDED_LIBS) \
		$(PIPS_LIBS)

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
