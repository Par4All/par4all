
TARGET	= step

PARSER_COMMENT = comment2pragma
PARSER_OMP = step_omp

INSTALL_INC = STEP_name.h STEP_RT_bootstrap.h STEP_RT_intrinsic.h

DERIVED_HEADERS = $(INSTALL_INC) $(PARSER_COMMENT).h $(PARSER_OMP).h ../../Runtimes/step/c/STEP.h
DERIVED_CFILES	= $(PARSER_COMMENT).c $(PARSER_OMP).c

INC_CFILES=	directives.c \
		analyse.c \
		compile_RT.c \
		compile.c

LIB_CFILES = $(DERIVED_CFILES) $(INC_CFILES)

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a


# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES)

clean: local-clean

local-clean:
	$(RM) $(DERIVED_CFILES) $(DERIVED_HEADERS) step_api.tmp steprt_f.h

$(PARSER_COMMENT).c: $(PARSER_COMMENT).l
	$(FLEX) $(LFLAGS) --prefix=$(PARSER_COMMENT)_ --header-file=$(PARSER_COMMENT).h -DYY_NO_INPUT -DIN_PIPS --outfile=`pwd`/$@ `pwd`/$^

$(PARSER_COMMENT).h: $(PARSER_COMMENT).c

$(PARSER_OMP).c: $(PARSER_OMP).l
	$(FLEX) $(LFLAGS) --prefix=$(PARSER_OMP)_ --header-file=$(PARSER_OMP).h -DYY_NO_INPUT --outfile=`pwd`/$@ `pwd`/$^

$(PARSER_OMP).h: $(PARSER_OMP).c

step_api.tmp: step_api.h
	grep '^[ \t]*extern[ \t]*void[ \t]*STEP_API' $^		 |	\
	sed 's/^[^(]*(step\([^)]*\))[ \t]*(\(.*\));/\1, \2/g'	 |	\
	sed 's/[ \t]*,[ \t]*/,/g' | sort > $@

STEP_name.h: STEP_name_variable.h step_api.tmp step_common.h
	cp STEP_name_variable.h $@ && \
	echo "/* Runtime MACRO (generated from step_common.h ) */" >> $@ && \
	grep "^#define[ \t]*STEP_" step_common.h |grep -v STEP_COMMON_H_ | sed 's/^#define[ \t]*\([^ \t]*\).*/#define \1_NAME "\1"/g' |sort >> $@ && \
	echo "/* Runtime MACRO (end) */" >> $@ && \
	awk 'BEGIN{FS=","; printf ("/* Runtime API intrinsic name (generated from step_api.h ) */\n") >> "$@"}\
	{\
	   base_name=$$1;\
	   RT_name="RT_STEP"base_name;\
	   printf ("#define %s \"%s\"\n", RT_name, toupper("step"base_name) ) >> "$@"\
	}\
	END{printf ("/* Runtime API intrinsic name (end) */\n") >> "$@"}' step_api.tmp

STEP_RT_bootstrap.h: step_api.tmp
	awk 'BEGIN{FS=","; printf ("/*\n\
	Runtime API intrinsic declaration (generated from step_api.h)\n\
	Included by src/Libs/bootstrap/bootstrap.c\n*/\n") > "$@"}\
	{\
	   base_name=$$1;\
	   RT_name="RT_STEP"base_name;\
	   nbargs=NF-1;\
	   if($$2=="void" || $$2=="")\
	      nbargs=0;\
	   else if($NF=="...")\
	      nbargs="(INT_MAX)";\
	   printf ("{%s, %s, overloaded_to_void_type, 0, 0},\n", RT_name, nbargs) >> "$@"\
	}\
	END{printf ("/* Runtime API intrinsic declaration (end) */\n") >> "$@"}' $^

STEP_RT_intrinsic.h: step_api.tmp
	awk 'BEGIN{FS=","; printf ("/*\n\
	Runtime API handler (generated from step_api.h )\n\
	Included by src/Libs/ri-utils/prettyprint.c\n*/\n") > "$@"}\
	{\
	   base_name=$$1;\
	   RT_name="RT_STEP"base_name;\
	   printf ("{%s, words_call_intrinsic, 0},\n", RT_name) >> "$@"\
	}\
	END{printf ("/* Runtime API handler (end) */\n") >> "$@"}' $^

../../Runtimes/step/c/STEP.h:
	$(MAKE) -C ../../Runtimes/step/c STEP.h
