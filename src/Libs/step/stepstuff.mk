
step_api.tmp: step_api.h
	grep '^[ \t]*extern[ \t]*void[ \t]*STEP_API' $^		 |	\
	sed 's/^[^(]*(step\([^)]*\))[ \t]*(\(.*\));/\1, \2/g'	 |	\
	sed 's/[ \t]*,[ \t]*/,/g' | sort > $@

STEP_name.h: STEP_name_variable.h step_api.tmp step_common.h
	cp $(srcdir)/STEP_name_variable.h $@ && \
	echo "/* Runtime MACRO (generated from step_common.h ) */" >> $@ && \
	grep "^#define[ \t]*STEP_" $(srcdir)/step_common.h |grep -v STEP_COMMON_H_ | sed 's/^#define[ \t]*\([^ \t]*\).*/#define \1_NAME "\1"/g' |sort >> $@ && \
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

# this one means that Runtimes stuff must be somehow compiled as well.
# so it must be am-ized...
$(top_srcdir)/src/Runtimes/step/c/STEP.h:
	$(MAKE) -C $(top_srcdir)/src/Runtimes/step/c STEP.h

# it seems that am does not handle several lex files in a directory
# so treating them manually is a simple workaround for that
PARSER_COMMENT	= comment2pragma
PARSER_OMP	= step_omp

$(PARSER_COMMENT).c: $(PARSER_COMMENT).lex
	$(FLEX) $(LFLAGS) --prefix=$(PARSER_COMMENT)_ --header-file=$(PARSER_COMMENT).h -DYY_NO_INPUT -DIN_PIPS --outfile=`pwd`/$@ $<

$(PARSER_COMMENT).h: $(PARSER_COMMENT).c

$(PARSER_OMP).c: $(PARSER_OMP).lex
	$(FLEX) $(LFLAGS) --prefix=$(PARSER_OMP)_ --header-file=$(PARSER_OMP).h -DYY_NO_INPUT --outfile=`pwd`/$@ $<

$(PARSER_OMP).h: $(PARSER_OMP).c
