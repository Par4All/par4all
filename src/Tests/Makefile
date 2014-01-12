# $Id$

LC3.inc	= ../../include
LC3.lib = ../../lib/LINUX_x86_64_LL

.PHONY: all
all: smith

# testing...
CC		= gcc -std=c99
CFLAGS		= -O2 -Wall
CPPFLAGS	= -I $(LC3.inc) -DLINEAR_VALUE_IS_LONGLONG
LDFLAGS		= -L $(LC3.lib)

%: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) -o $@

F.c	= $(wildcard *.c)
F.o	= $(F.c:%.c=%.o)
F.exe	= $(F.c:%.c=%)

MX.lib	= -lmatrix -larithmetique
SC.lib	= -lsc -lcontrainte -lvecteur -larithmetique
PO.lib	= -lpolyedre -lsg

smith: LDFLAGS += $(MX.lib)
feasability: LDFLAGS += $(SC.lib)
#sc_env: LDFLAGS += $(PO.lib) $(SC.lib) $(MX.lib)

clean:
	$(RM) *~ $(F.o) $(F.exe)
