FC=gfortran
PAR_FLAG=-fopenmp
OPT_FLAG=-O2
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
	rm -rf *.o *.database *~
	rm -f matmul
	rm -f matmul_par
