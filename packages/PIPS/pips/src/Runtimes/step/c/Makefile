#********************************************************************************
#                                                                               *
#     Authors       :   Abdellah. Kouadri                                       *
#                       Daniel Millot                                           *		
#                       Frédérique Silber-Chaussumier                           *
#                                                                               *
#     Date          :   23/10/2009                                              *
#                                                                               *
#     File          :   Makefile                                                *
#                                                                               *
#     Version       :   1.1                                                     *
#                                                                               *  
#     Description   :   Builds the runtime static library                       *  
#                       target test builds a simple autotest                    *
#                                                                               *  
# *******************************************************************************

CC           	= mpicc
LIBRARY      	= libstep.a
AUTO_TEST    	= test
INCLUDE_DIR  	= ./util
LD_FLAGS     	= -L./util -ltimings 
CFLAGS       	= -g -O2




SRC          	= steprt.c steprt_private.c steprt_comm_alg.c steprt_wrapper.c
OBJ          	= $(SRC:.c=.o)

all:            $(LIBRARY) 

%.o: 		%.c
		$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ -c $< 


$(AUTO_TEST):   $(SRC)
	make -C ./util libs
		$(CC) -DTEST $(CFLAGS) -I$(INCLUDE_DIR) $^ -o $@ $(LD_FLAGS) 

 
$(LIBRARY):     $(OBJ)                                   
		ar rcs $@ $^


clean:
		@rm -f *.o  *~ 2> /dev/null
		@rm -f $(LIBRARY) $(AUTO_TEST)  2> /dev/null 


