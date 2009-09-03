FC=gfortran
PAR_FLAG=-fopenmp
PIPS=tpips

all: parallel sequential

parallel: matmul1_par matmul2_par

sequential: matmul1 matmul2

matmul1:
	$(FC) matmul1.f -o matmul1

matmul1_par:
	$(PIPS) matmul1.tpips
	$(FC) $(PAR_FLAG) matmul1.database/Src/matmul1.f -o matmul1_par

matmul2:
	$(FC) matmul2.f -o matmul2

matmul2_par:
	$(PIPS) matmul2.tpips
	$(FC) $(PAR_FLAG) matmul2.database/Src/matmul2.f -o matmul2_par

clean:
	rm -rf *.o *.database *~
	rm -f matmul1 matmul2
	rm -f matmul1_par matmul2_par
