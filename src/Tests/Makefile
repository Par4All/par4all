# $Id$

LC3.inc	= ../../include
LC3.lib = ../../lib/LINUX_x86_64_LL

.PHONY: all
all: smith

# testing...
CC		= gcc
CFLAGS		= -O2 -Wall
CPPFLAGS	= -I $(LC3.inc)
LDFLAGS		= -L $(LC3.lib) -lmatrix -larithmetique

%: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) -o $@

F.c	= $(wildcard *.c)
F.o	= $(F.c:%.c=%.o)
F.exe	= $(F.c:%.c=%)

clean:
	$(RM) *~ $(F.o) $(F.exe)
