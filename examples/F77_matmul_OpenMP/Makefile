FC=gfortran
PAR_FLAG=-fopenmp
OPT_FLAG=-O3
PIPS=tpips

all: parallel sequential

parallel: matmul_par

sequential: matmul

matmul:
	$(FC) $(OPT_FLAG) matmul.f -o matmul

matmul_par:
	./PipsCheck.sh
	$(PIPS) matmul.tpips
	$(FC) $(PAR_FLAG) $(OPT_FLAG) matmul.database/Src/matmul.f -o $@

clean:
	rm -rf *.database p4a_new_files *.build
	rm -f *.o *~ *.mod *.p4a.f
	rm -f matmul
	rm -f matmul_par
