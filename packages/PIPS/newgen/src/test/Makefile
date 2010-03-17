# $Id$

all: reserv

ROOT	= ../..
include $(ROOT)/makes/main.mk

local:	test_hash test_reserv

test_hash: hash_test
	@echo Warning: do not make reserv \(GO\)
	./hash_test 100

spec.h: newres.newgen
	$(EXE.d)/newgen -c $< | $(EXE.d)/make_all_specs > $@

%.h: %.newgen
	$(EXE.d)/newgen -c $< | $(EXE.d)/make_all_specs > $@

hash_test: hash_test.c
	$(COMPILE) $< -o $@

lieu.c: spec.h lieu.h
newres.c: spec.h newres.h

reserv: reserv.o lieu.o newres.o
	$(LD) -L$(LIB.d) $(LDFLAGS) reserv.o lieu.o newres.o -lgenC -o $@

reserv.o: spec.h lieu.h newres.h

%.o: %.c
	$(CC) $(CFLAGS)  -I. -I$(INC.d) $(CPPFLAGS) -c $<

reserv: spec.h lieu.h

test_reserv: reserv
	reserv < newres.in
	diff sortie sortie2;

clean:
	$(RM) hash_test reserv spec.h lieu.h lieu.spec newres.h newres.spec \
		lieu.c newres.c sortie sortie[234] core *~ *.o
