# $Id$

OBSOLETE_FILES= elarg.c elim_red.c env.c incl_p_h.c inter_poly.c \
	poly.c sc_elarg.c inter_demi.c inter_hyp.c inter_obj.c saturation.c \
	saturation.h sc_to_sg.c syst_convert.c liste-table.h GENPGM_TAGS.h

TARGET	= polyedre

LIB_CFILES= 	sc_enveloppe.c chernikova.c 

LIB_TARGET	= lib$(TARGET).a

INC_TARGET	= $(TARGET).h

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
PROJECT = linear
include $(ROOT)/makes/main.mk
