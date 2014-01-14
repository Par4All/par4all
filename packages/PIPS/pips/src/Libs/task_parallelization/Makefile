TARGET	= task_parallelization

LIB_CFILES=	\
		SDG.c\
		cost_model.c\
		BDSC.c\
		HBDSC.c\
		instrumentation.c\
		spire_generation.c\
		communication_generation.c\
		openmp_generation.c\
		mpi_generation.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
