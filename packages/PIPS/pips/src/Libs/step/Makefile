
TARGET	= step

LIB_CFILES=	step.c \
		atomizer.c \
		outlining.c \
		directives.c \
		directive_parallel.c \
		directive_do.c \
		directive_section.c \
		directive_reduction.c \
		directive_master.c \
		directive_critical.c \
		directive_barrier.c \
		directive_private.c \
		directive_transformation.c \
		analyse.c \
		compile_region.c \
		compile.c \
		compile_RT.c \
		compile_parallel.c \
		compile_do.c \
		compile_reduction.c \
		compile_master.c \
		compile_critical.c \
		compile_barrier.c \
		compile_private.c \
		compile_share.c 

INC_TARGET =    $(TARGET).h

INSTALL_INC = STEP_name.h STEP_RT_bootstrap.h STEP_RT_intrinsic.h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
