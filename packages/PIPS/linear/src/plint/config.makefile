#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/09 16:56:18 $, 

LIB_CFILES=	find-eg.c \
		pldual.c \
		plfonct-eco.c \
		plgomory.c \
		plint.c \
		plpivoter.c \
		plreal.c \
		plsimplexe.c \
		plsolution.c \
		plsommet-op.c \
		plsomvb-test.c \
		plvar-ecart.c \
		plvbase.c \
		sc-fais-int-sm.c \
		sc-fais-int.c \
		sc-red-int.c \
		sc-res-smith.c \
		sc_to_matrice.c

LIB_HEADERS=	plint-local.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 

# end of $RCSfile: config.makefile,v $
#
