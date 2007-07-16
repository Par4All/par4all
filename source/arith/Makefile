
#########################################################################
# WARNING :                                                             #
# macro EXTRA_FLAGS must have been set when calling                     #
# this make (if not set, default is 32 bits integers)                   #
# OBJ_DIR must be set to the corresponding directory of objects         #
#########################################################################

# Parent makefile sets these values

#  Choose one of these :
#  LINEAR_VALUE_IS_LONGLONG
#  LINEAR_VALUE_IS_LONG
#  LINEAR_VALUE_IS_FLOAT
#  LINEAR_VALUE_IS_CHARS   (used for type checking)
#  LINEAR_VALUE_IS_INT     
#  GNUMP (default)
#EXTRA_FLAGS = -DLINEAR_VALUE_IS_LONGLONG -DLINEAR_VALUE_PROTECT_MULTIPLY \
#  -DLINEAR_VALUE_ASSUME_SOFTWARE_IDIV
#OBJ_DIR = Obj.$(OSTYPE)

include ../vars.mk

#CC = gcc
#CFLAGS = -O4 -Wall -g 
#RANLIB = ranlib


LIB_CFILES=  errors.c

HEADERS=  	assert.h arithmetic_errors.h arithmetique.h

LIB_OBJECTS= $(LIB_CFILES:%.c=$(OBJ_DIR)/%.o)


$(OBJ_DIR)/arithmetique.a:: $(OBJ_DIR) $(LIB_OBJECTS) Makefile
	ar r $(OBJ_DIR)/arithmetique.a $(LIB_OBJECTS)
	$(RANLIB) $(OBJ_DIR)/arithmetique.a

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(LIB_OBJECTS): $(HEADERS) $(LIB_CFILES) Makefile
	$(CC) -c $(CFLAGS) $(*:$(OBJ_DIR)/%=%.c) -o $*.o

clean:
	-rm -rf Obj.*



