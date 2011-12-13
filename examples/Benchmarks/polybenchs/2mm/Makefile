LOCAL_TARGET=2mm


include ../bench.mk

ifdef SMALL_INPUT
SIZE_PARAMS=-DNI=64 -DNJ=64 -DNK=64 -DNL=64
else 
ifndef std_size 
SIZE_PARAMS=-DNI=2048 -DNJ=2048 -DNK=2048 -DNL=2048
endif
endif	

CPPFLAGS+=$(SIZE_PARAMS)
