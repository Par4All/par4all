step_api.tmp: step_api.h 
	grep '^[ \t]*extern[ \t]*void[ \t]*STEP_API' $^		 |	\
	sed 's/^[^(]*(step\([^)]*\))[ \t]*(\(.*\));/\1, \2/g'	 |	\
	sed 's/[ \t]*,[ \t]*/,/g' | sort > $@

STEP_name.h: STEP_name_variable.h step_api.tmp step_common.h Makefile
	cp $(srcdir)/STEP_name_variable.h $@ && chmod u+w $@ && \
	echo "/* Runtime MACRO (generated from step_common.h ) */" >> $@ && \
	grep "^#define[ \t]*STEP_" $(srcdir)/step_common.h |grep -v STEP_COMMON_H_ | sed 's/^#define[ \t]*\([^ \t(]*\).*/#define \1_NAME "\1"/g' |sort >> $@ && \
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
	Runtime API handler (generated from step_api.h )\n*/\n") > "$@"}\
	{\
	   base_name=$$1;\
	   RT_name="RT_STEP"base_name;\
	   printf ("{%s, { words_call_intrinsic, 0} },\n", RT_name) >> "$@"\
	}\
	END{printf ("/* Runtime API handler (end) */\n") >> "$@"}' $^


