# Sur jaguar :
# MPIDIR = /opt/openmpi		# pour OpenMPI, et mpirun avec la syntaxe :
#     mpirun --mca btl_tcp_if_include eth0 -machinefile machines.txt -np 2 exe
#		pour utiliser le fichier machines.txt (sinon par defaut,
#          			utilise uniquement le noeud de lancement)
# MPIDIR = /opt/mpich/gnu	# pour MPICH, et mpirun avec la syntaxe :
#          mpirun -np 10 testmpitimings
#		(machinefile par defaut : /opt/mpich/gnu/share/machines.LINUX)
# MPIDIR = /opt/mpich2/gnu	# pour MPICH2, et mpiexec
# MPIBINDIR = $(MPIDIR)/bin

OMPIBINDIR=/opt/openmpi/bin
OMPIDEVBINDIR=/opt/openmpi-dev/bin
MPICHBINDIR=/opt/mpich/gnu/bin
MPICH2BINDIR=/opt/mpich2/gnu/bin

MACHINEFILE = /opt/mpich/gnu/share/machines.LINUX
TMPFILE = /tmp/myTempMachineFile
NBPROC = 4

PROG = hello

all:
	@echo choisir une cible correspondant a l\'une des versions de MPI :
	@echo -e "\t ompi, mpich, ou mpich2"
	@echo ou la cible cleanall

ompi:$(PROG).ompi
	@echo ;echo -ne ">>>\t";echo -n "execution avec OpenMPI ($(OMPIBINDIR))";echo -e "\t<<<";echo
	$(OMPIBINDIR)/mpirun --mca btl_tcp_if_include eth0 -machinefile $(MACHINEFILE) -np $(NBPROC) $(PROG).ompi

$(PROG).ompi:$(PROG).c
	$(OMPIBINDIR)/mpicc $(PROG).c -o $(PROG).ompi

mpich:$(PROG).mpich
	@echo ;echo -ne ">>>\t";echo -n execution avec mpich;echo -e "\t<<<";echo
	$(MPICHBINDIR)/mpirun -np $(NBPROC) $(PROG).mpich

$(PROG).mpich:$(PROG).c
	$(MPICHBINDIR)/mpicc $(PROG).c -o $(PROG).mpich

mpich2:$(PROG).mpich2 mpdboot
	@echo ;echo -ne ">>>\t";echo -n execution avec mpich2;echo -e "\t<<<";echo
	wdir=`pwd` ; $(MPICH2BINDIR)/mpiexec -l -np $(NBPROC) $$wdir/$(PROG).mpich2
	@echo; make mpdclear

$(PROG).mpich2:$(PROG).c
	$(MPICH2BINDIR)/mpicc $(PROG).c -o $(PROG).mpich2

mpdboot :
	@$(MPICH2BINDIR)/mpdallexit 2>&1 >/dev/null;\
	$(MPICH2BINDIR)/mpd --daemon;\
	FILE=$(TMPFILE)`logname`.txt;\
	cat $(MACHINEFILE)|head -$(NBPROC) > $$FILE;\
	mpdportId=`$(MPICH2BINDIR)/mpdtrace -l | cut -d' ' -f1 | cut -d_ -f2`;\
	for iNode in `cat $$FILE` ; do ssh $$iNode "$(MPICH2BINDIR)/mpd --daemon --host=`hostname` --port=$$mpdportId"; done

mpdclear :
	@$(MPICH2BINDIR)/mpd &
	@$(MPICH2BINDIR)/mpdallexit 2>&1 /dev/null;\
	FILE=$(TMPFILE)`logname`.txt;\
	wdir=`pwd` ; ( $(MPICH2BINDIR)/mpdcleanup --file=$$FILE ) 2>&1 > /dev/null;\
	/bin/rm -f $$FILE

cleanall:
	rm -rf *~ *.o $(PROG).ompi $(PROG).mpich $(PROG).mpich2

# Pour debug :
#	mpirun -np 4 xterm -e gdb executable