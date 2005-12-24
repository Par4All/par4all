#
#
#   	WP65: PUMA ESPRIT PROJECT 2701
#   	------------------------------
#
# Corinne Ancourt, Francois Irigoin, Lei Zhou	     17 October 1991
#
####### The source files directly involved in wp65 are:

TARGET	= wp65

LIB_CFILES =	code.c tiling.c variable.c instruction_to_wp65_code.c \
		wp65.c basis.c \
		find_iteration_domain.c model.c references.c communications.c 

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
