# $Id$

all: reserv

ifdef NEWGEN_ROOT
ROOT	= $(NEWGEN_ROOT)
else
ROOT	= ../..
endif

include $(ROOT)/makes/main.mk

local:	test_hash test_reserv

test_hash: hash_test
	@echo Warning: do not make reserv \(GO\)
	./hash_test 100

spec.h: newres.newgen
	$(SHR.d)/newgen -c $< | $(SHR.d)/make_all_specs > $@

hash_test: hash_test.c
	$(COMPILE) $< -o $@

reserv: reserv.c 
	$(CC) $(CFLAGS)  -I. -I$(INC.d) $(CPPFLAGS) \
		-L$(LIB.d) $(LDFLAGS) -lgenC $<  -o $@

reserv: spec.h lieu.h

test_reserv: reserv
	reserv < newres.in ; 
	diff sortie sortie2;

clean: 
	$(RM) hash_test reserv spec.h lieu.h lieu.spec newres.h newres.spec \
		lieu.c newres.c sortie sortie[234] core *~
