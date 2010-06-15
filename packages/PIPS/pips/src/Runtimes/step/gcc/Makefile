TARGET = libstep.a

CSOURCES = step.c

#CC = gcc-4.2
CC = gcc

CCFLAGS = -g -O2  

#MPIDIR = /usr/local/openmpi3.1
MPIDIR = /usr/lib/mpich

LOCALINC= -I. -I./include -I./util
INCLDIR = $(LOCALINC) -I/usr/include -I/usr/local/include -I$(MPIDIR)/include

LIBDIR = -L/usr/local/lib -L$(MPIDIR)/lib -L ./util
LFLAGS = -lm -lmpi -lutil
LTRACEFLAGS = $(LFLAGS) -ltimings

SOURCEFILES = $(CSOURCES:=.c)
OBJECTFILES = $(CSOURCES:=.o) 


all:		$(TARGET) steptest steptrace

.c.o:		
		$(CC) $(CFLAGS) $(INCLDIR) -o $@ -c $<

steptest:	step.c  
		$(CC) $(CFLAGS) $(INCLDIR) $(LIBS) -D TEST_ $< -o $@ $(LFLAGS)

tracesteptest:	step.c  
		$(CC) $(CFLAGS) $(INCLDIR) $(LIBS) -D TEST_ -DTRACE $< -o $@ $(LTRACEFLAGS)

libstep.a:	step.o 
		ar rcs $(TARGET) $^

steptrace: step.c
		vtcc -vt:cc $(CC) $(INCLDIR) $(LOCALINC) $(CFLAGS) $(LIBS) $< -lutil -D TEST_ -DVTRACE -D __arch64__ -o $@   -m64 -lsz -lz -lmpi  


clean:
		rm -f Makefile.bak *.o *.a *~ steptest

