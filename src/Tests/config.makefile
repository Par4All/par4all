# $RCSfile: config.makefile,v $ (Version $Revision$)
# $Date: 1996/10/10 18:40:22 $, 
#
# revu par FC
# pour fabriquer un executable FOO, faire "make FOO"
# utilise par defaut les librairies linear installees.
# utilise les librairies locales en priorite.
# 

CFILES	=	sc_to_sg_test1.c \
		sc_to_sg_test.c \
		env_test.c \
		elarg_test.c \
		sc_min.c \
		simp.c \
		test_chernikova.c \
		test_env_chernikova.c \
		sc_env.c \
		time_sg_union.c \
		feasability.c

SOURCES	= $(CFILES) 

all: test_chernikova simp

%: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $< $(LINEAR_LIBS) -o $@

# test_simp:	simp
# 	sh ./test_simp.sh | sed -f filtre.sed > resultat.2 ; 
# 	diff resultat.2 RM/resultat.1 
# 	$(RM) resultat.2

clean: local-clean
local-clean:
	$(RM) $(CFILES:.c=) *~

# end of $RCSfile: config.makefile,v $
#
