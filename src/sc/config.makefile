#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 2003/07/28 09:17:51 $, 

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
		sc_debug.c \
		sc_janus_feasibility.c \
		isolve.c \
		r1.c

LIB_HEADERS=	sc-local.h \
		sc-private.h \
		sc_gram.y \
		sc_lex.l \
		iabrev.h \
		iproblem.h \
		rproblem.h

DERIVED_HEADERS=sc_gram.h
DERIVED_CFILES=	sc_gram.c sc_lex.c

# $(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

LIB_OBJECTS= $(LIB_CFILES:.c=.o) $(DERIVED_CFILES:.c=.o) 

YY2SYST	= sed '/extern char \*malloc/d;/^FILE *\*/s,=[^,;]*,,g;\
	s/YY/SYST_/g;s/yy/syst_/g;' 

sc_lex.c: sc_lex.l
	$(SCAN) $< | $(YY2SYST)	> $@

sc_gram.c sc_gram.h: sc_gram.y
	$(PARSE) -d $<
	$(YY2SYST) y.tab.c > sc_gram.c
	$(YY2SYST) y.tab.h > sc_gram.h
	$(RM) y.tab.c y.tab.h
 
# end of $RCSfile: config.makefile,v $
#
