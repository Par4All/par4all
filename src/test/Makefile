#
# Makefile for Test
#

all: reserv

ARCH=$(PIPS_ARCH)
ROOT=$(NEWGEN_ROOT)

include $(ROOT)/Include/makefile_macros.$(ARCH)

DEV=$(NEWGEN_DEVEDIR)
INC=$(NEWGEN_DEVEDIR)/genC
LIB=$(DEV)/genC/$(ARCH)

#
# test the local version

local:	test_hash test_reserv

test_hash: hash_test
	@echo Warning: do not make reserv \(GO\)
	./hash_test 100

#
# 

spec.h: newres.newgen
	$(DEV)/scripts/newgen -c $< | $(DEV)/scripts/make_all_specs > $@

reserv: reserv.c
	$(CC) $(CFLAGS)  -I$(INC) $(CPPFLAGS) \
		-L$(LIB) $(LDFLAGS) $< -lgenC -o $@

reserv: spec.h $(LIB)/libgenC.a

test_reserv: reserv
	reserv < newres.in ; 
	diff sortie sortie2;

clean: 
	$(RM) hash_test reserv spec.h lieu.h lieu.spec newres.h newres.spec \
		sortie sortie[234] core

#
# That is all
#
