#
# $Id$
#

# -DNO_MESSAGES

OBSOLETE_FILES= affect.c elarg.c elim_red.c env.c incl_p_h.c inter_poly.c \
	poly.c sc_elarg.c inter_demi.c inter_hyp.c inter_obj.c saturation.c \
	saturation.h sc_to_sg.c syst_convert.c liste-table.h GENPGM_TAGS.h

LIB_CFILES= 	sc_enveloppe.c chernikova.c 

LIB_HEADERS=	polyedre-local.h \
		$(OBSOLETE_FILES)

LIB_OBJECTS= $(LIB_CFILES:.c=.o)
