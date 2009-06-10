#####
lib_LIBRARIES = libgenC.a

CPROTO_GEN=newgen_auto_string.h
BUILT_SOURCES = $(CPROTO_GEN) read.h gram.h

AM_YFLAGS= -d

headers=newgen_hash.h newgen_types.h newgen_include.h genC.h newgen_list.h newgen_array.h newgen_set.h newgen_generic_mapping.h newgen_string_buffer.h newgen_generic_function.h newgen_assert.h newgen_generic_stack.h newgen_map.h newgen_stack.h $(CPROTO_GEN)

sources=build.c genC.c genClib.c tabulated.c hash.c set.c list.c stack.c array.c string.c string_buffer.c gram.y token.l read.y read_token.l

libgenC_a_SOURCES=$(sources) $(headers) 



newgen_auto_string.h:string.c
	$(CPROTO) -evcf2 -X 0 -E "$(CPP) $(CPPFLAGS)" -o $@ $<


#####

bin_PROGRAMS= newC
newC_SOURCES=new.c 
newC_LDADD=./libgenC.a

