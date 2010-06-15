# Sur jaguar :
# MPIDIR = /opt/openmpi		# pour OpenMPI, et mpirun avec la syntaxe :
#     mpirun --mca btl_tcp_if_include eth0 -machinefile machines.txt -np 2 exe
#		pour utiliser le fichier machines.txt (par defaut,
#          			utilise uniquement le noeud de lancement)
# MPIDIR = /opt/mpich/gnu	# pour MPICH, et mpirun avec la syntaxe :
#          mpirun -np 10 testmpitimings
#		(machinefile par defaut : /opt/mpich/gnu/share/machines.LINUX)
# MPIDIR = /opt/mpich2/gnu	# pour MPICH2, et mpiexec
# MPICC = $(MPIDIR)/bin/mpicc
# /opt/openmpi/bin/mpirun --mca btl_tcp_if_include eth0 -machinefile $(MACHINEFILE) -np $(NBPROC) $(PROG).ompi
# /opt/mpich/gnu/bin/mpirun -np $(NBPROC) $(PROG).mpich
#
# /opt/mpich2/gnu/bin/mpiexec -l -np $(NBPROC) $$wdir/$(PROG).mpich2

ifeq ($(MPI),ompi)
	MPIDIR=/opt/openmpi
	MPIRUN=/opt/openmpi/bin/mpirun --mca btl_tcp_if_include eth0 -machinefile $(MACHINEFILE) -np $(NBPROC)
endif

ifeq ($(MPI),mpich)
	MPIDIR=/opt/mpich/gnu
	MPIRUN=/opt/mpich/gnu/bin/mpirun -np $(NBPROC)
endif

ifeq ($(MPI),mpich2)
	MPIDIR=/opt/mpich2/gnu
	MPIRUN=echo; echo A completer pour executer
endif

CC = gcc
MPICC = $(MPIDIR)/bin/mpicc

INCDIR = -I.
LIBDIR = -L.

CFLAGS = -Wall
CFLAGS += $(INCDIR)
LDFLAGS = -lutil

MACHINEFILE = /opt/mpich/gnu/share/machines.LINUX
NBPROC = 4

help:
	@echo Definir la macro MPI pour choisir entre ompi, mpich et mpich2

all: libs tests

libs: libutil.a libtimings.a libthread_timings.a

tests: testtrace testtimings testutil testmpitimings testthreadtimings

testtrace: trace.c trace.h
	$(CC) $(CFLAGS) -DTEST $< -o $@ $(LIBDIR) -ltimings

testtimings: timings.c timings.h
	$(CC) $(CFLAGS) -DTEST $< -o $@ $(LIBDIR) $(LDFLAGS)

testutil: util.c util.h
	$(CC) $(CFLAGS) -DTEST $< -o $@ $(LIBDIR) $(LDFLAGS)

testmpitimings.$(MPI): concurrent_timings.c concurrent_timings.h libtimings.a libutil.a
	$(MPICC) $(CFLAGS) -DTRACE -DMPI -DTEST -DDEBUG $< -o $@ $(LIBDIR) $(LDFLAGS) -ltimings

testthreadtimings: concurrent_timings.c concurrent_timings.h
	$(CC) $(CFLAGS) -DTRACE -DTHREAD -DTEST $< -o $@ $(LIBDIR) $(LDFLAGS) -ltimings -lpthread

run: testmpitimings.$(MPI)
	$(MPIRUN) testmpitimings.$(MPI)

%.o: %.c %.h

trace.o: trace.h trace.c

timings.o: timings.h timings.c

libutil.a: trace.o util.o
	$(AR) r $@ $^

libtimings.a: timings.o	
	$(AR) r $@ $^



libmpi_timings.a: mpi_timings.o 
	ar r $@ $^

libthread_timings.a: thread_timings.o 
	ar r $@ $^

mpi_timings.o: concurrent_timings.c concurrent_timings.h
	$(MPICC) $(CFLAGS) -DMPI -o $@ $< -c

thread_timings.o: concurrent_timings.c concurrent_timings.h
	$(CC) $(CFLAGS) -DTHREAD -o $@ $< -c

clean:
	rm -f *~ *.o

cleantest:
	rm -f testtrace testtimings testutil testthreadtimings testmpitimings.ompi testmpitimings.mpich testmpitimings.mpich2

cleanall:clean cleantest
	rm -f *.a
