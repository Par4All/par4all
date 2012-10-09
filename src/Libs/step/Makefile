
TARGET	= step

PARSER_COMMENT = comment2pragma

LEXER_STEP = step_lexer
PARSER_STEP = step_bison_parser


INSTALL_INC = STEP_name.h STEP_RT_bootstrap.h STEP_RT_intrinsic.h

DERIVED_HEADERS = $(INSTALL_INC) $(PARSER_COMMENT).h $(LEXER_STEP).h $(PARSER_STEP).h ../../Runtimes/step/c/STEP.h
DERIVED_CFILES	= $(PARSER_COMMENT).c $(LEXER_STEP).c $(PARSER_STEP).c

INC_CFILES=	directives.c \
		analyse.c \
		compile_RT.c \
		compile_mpi.c \
		compile_regions.c \
		compile.c \
		parser.c \
		install.c

LIB_CFILES = $(DERIVED_CFILES) $(INC_CFILES)

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a


# common stuff
ROOT    = ../../..
PROJECT = pips
CPPFLAGS += -DPIPS_RUNTIME_DIR=\"$(CURDIR)/../../Runtimes\"
include $(ROOT)/makes/main.mk

# local stuff
$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES)

clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) step_api.tmp steprt_f.h

$(PARSER_COMMENT).c: $(PARSER_COMMENT).l
	$(FLEX) $(LFLAGS) --prefix=$(PARSER_COMMENT)_ --header-file=$(PARSER_COMMENT).h -DYY_NO_INPUT -DIN_PIPS --outfile=`pwd`/$@ `pwd`/$^

$(PARSER_COMMENT).h: $(PARSER_COMMENT).c

$(LEXER_STEP).c: $(LEXER_STEP).l
	$(FLEX) $(LFLAGS) --prefix=step_ --header-file=$(LEXER_STEP).h -DYY_NO_INPUT --outfile=`pwd`/$@ `pwd`/$^

$(PARSER_STEP).c: $(PARSER_STEP).y
	$(BISON) -d --name-prefix=step_ $(PARSER_STEP).y -o $(PARSER_STEP).c

$(PARSER_STEP).h: $(PARSER_STEP).c

srcdir=.
include stepstuff.mk

../../Runtimes/step/c/STEP.h:
	$(MAKE) -C ../../Runtimes/step/c STEP.h
