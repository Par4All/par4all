#
# $RCSfile: config.makefile,v $ for make
#

SCRIPTS = 	MakeExtern \
		make-builder-map \
		make-extern \
		make-extern.etags \
		make-includes \
		make-install-pips \
		make-pips \
		make-pips-menu \
		makedep \
		makemake \
		install-pips-src \
		tape-pips \
		make-gdbinit

SFILES=		mkextern.l
RFILES=		mkextern
FILES =		ctags2extern.awk

mkextern: mkextern.l
	$(LEX) mkextern.l
	$(CC) $(CFLAGS) -o mkextern lex.yy.c
	strip mkextern
	$(RM) lex.yy.c

# that is all
#
