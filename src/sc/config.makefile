#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/24 11:00:33 $, 

LIB_CFILES=	sc_alloc.c \
	        sc_projection.c \
	        sc_read.c \
	        sc.c \
	        sc_integer_analyze.c \
	        sc_feasibility.c \
	        sc_intersection.c \
	        sc_integer_projection.c \
	        sc_normalize.c \
	        sc_build_sc_nredund.c \
	        sc_oppose.c \
	        sc_triang_elim_redond.c \
	        sc_elim_redund.c \
	        sc_elim_simple_redund.c \
		sc_insert_eq.c \
		sc_transformation.c\
		sc_var.c \
	        sc_eval.c \
		sc_unaires.c \
		sc_error.c \
		sc_io.c \
		sc_new_loop_bound.c \
		sc_simplexe_feasibility.c \
		sc_debug.c

LIB_HEADERS=	sc-local.h \
		sc-private.h \
		sc_gram.y \
		sc_lex.l

DERIVED_HEADERS=y.tab.h
DERIVED_CFILES=	sc_gram.c sc_lex.c

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

LIB_OBJECTS= $(LIB_CFILES:.c=.o) $(DERIVED_CFILES:.c=.o) 

sc_lex.c: sc_lex.l
	$(SCAN) $< | sed 's/YY/SC_/g;s/yy/sc_/g;' > $@

sc_gram.c y.tab.h: sc_gram.y
	$(PARSE) -d $<
	sed -e '/extern char \*malloc/d;s/YY/SC_/g;s/yy/sc_/g;' \
		y.tab.c > sc_gram.c
	$(RM) y.tab.c
 
# end of $RCSfile: config.makefile,v $
#
