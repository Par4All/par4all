#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1997/04/14 10:57:30 $, 

# -DNO_MESSAGES

CPPFLAGS+=	-DCHECK_OVERFLOW 

OBSOLETE_FILES= affect.c elarg.c elim_red.c env.c incl_p_h.c inter_poly.c \
	poly.c sc_elarg.c inter_demi.c inter_hyp.c inter_obj.c saturation.c \
	saturation.h sc_to_sg.c syst_convert.c liste-table.h GENPGM_TAGS.h

LIB_CFILES= 	sc_enveloppe.c \
		chernikova.c 

# from IRISA:
OTHER_CFILES=	polyhedron.c \
		vector.c

LIB_HEADERS=	polyedre-local.h \
		types-irisa.h \
		vector.h \
		polyhedron.h \
		$(OTHER_CFILES) \
		$(OBSOLETE_FILES)

LIB_OBJECTS= $(LIB_CFILES:.c=.o) $(OTHER_CFILES:.c=.o)
 
# end of $RCSfile: config.makefile,v $
#
