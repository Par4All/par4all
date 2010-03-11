TARGET = phrase
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=smalltalk-defs.h fsm_generation.h phrase_tools.h spaghettify.h phrase_distribution.h comEngine.h comEngine_generate_code.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libphrase.la
libphrase_la_SOURCES=comEngine_optimization_function.c \
                comEngine_HRE_distribute.c \
                comEngine_generate_code.c \
                phrase_remove_dependences.c \
                comEngine_generate_HRECode.c \
                comEngine_generate_procCode.c \
                comEngine_feasability.c \
                comEngine_distribute.c \
                phrase_distributor.c \
		phrase_distributor_control_code.c \
		phrase_distributor_communication.c \
		distribution_context.c \
		fsm_generation.c \
		print_code_smalltalk.c \
		fsm_merge_states.c \
		fsm_split_state.c \
		full_fsm_generation.c \
		loop_spaghettify.c \
		whileloop_spaghettify.c \
		forloop_spaghettify.c \
		test_spaghettify.c \
		spaghettify.c \
		full_spaghettify.c \
		fsm_tools.c \
		phrase_tools.c


include $(srcdir)/../pipslibs_includes.mk
