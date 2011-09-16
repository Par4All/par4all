# $Id$
ROOT    = ../../..

TARGET	= step

PARSER_COMMENT = comment2pragma
PARSER_OMP = step_omp

INSTALL_INC = STEP_name.h STEP_RT_bootstrap.h STEP_RT_intrinsic.h

DERIVED_HEADERS = $(INSTALL_INC) $(PARSER_COMMENT).h $(PARSER_OMP).h $(ROOT)/src/Runtimes/step/c/STEP.h
DERIVED_CFILES	= $(PARSER_COMMENT).c $(PARSER_OMP).c

INC_CFILES=	directives.c \
		analyse.c \
		compile_RT.c \
		compile.c

LIB_CFILES = $(DERIVED_CFILES) $(INC_CFILES)

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a


# common stuff
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES)

clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) step_api.tmp steprt_f.h

srcdir	= .
top_srcdir	= $(ROOT)
include stepstuff.mk
