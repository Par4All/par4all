#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/09 13:30:17 $, 

CPPFLAGS+=	-DCHECK_OVERFLOW -DNO_MESSAGES

LIB_CFILES=	affect.c \
		elarg.c \
		elim_red.c \
		env.c \
		incl_p_h.c \
		inter_demi.c \
		inter_hyp.c \
		inter_obj.c \
		inter_poly.c \
		poly.c \
		saturation.c \
		sc_to_sg.c \
		syst_convert.c \
		sc_enveloppe.c \
		sc_elarg.c \
		chernikova.c 

OTHER_CFILES=	polyhedron.c \
		vector.c

LIB_HEADERS=	polyedre-local.h \
		saturation.h \
		liste-table.h \
		GENPGM_TAGS.h \
		types-irisa.h \
		vector.h \
		polyhedron.h \
		$(OTHER_CFILES)

LIB_OBJECTS= $(LIB_CFILES:.c=.o) $(OTHER_CFILES:.c=.o)
 
# end of $RCSfile: config.makefile,v $
#
