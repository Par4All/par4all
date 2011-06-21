TOP := $(dir $(lastword $(MAKEFILE_LIST)))

TARGETS=2mm 3mm adi atax bicg correlation covariance doitgen dynprog fdtd-2d fdtd-apml gauss-filter gemm gemver gesummv gramschmidt jacobi-1d-imper jacobi-2d-imper lu ludcmp mvt seidel symm-exp syr2k syrk trisolv trmm

default:
	more $(TOP)/USAGE

clean: 
	for target in $(TARGETS) ; do \
		make -C $$target clean ; \
	done

dist-clean: 
	for target in $(TARGETS) ; do \
		make -C $$target dist-clean ; \
	done


run_seq:
	for target in $(TARGETS) ; do \
		make -C $$target run_seq ; \
	done

seq:
	for target in $(TARGETS) ; do \
		make -C $$target $@ ; \
	done

run_openmp:
	for target in $(TARGETS) ; do \
		make -C $$target run_openmp ; \
	done

openmp:
	for target in $(TARGETS) ; do \
		make -C $$target $@ ; \
	done


run_cuda:
	for target in $(TARGETS) ; do \
		make -C $$target run_cuda ; \
	done

cuda:
	for target in $(TARGETS) ; do \
		make -C $$target $@ ; \
	done

run_cuda_opt:
	for target in $(TARGETS) ; do \
		make -C $$target run_cuda_opt ; \
	done

cuda_opt:
	for target in $(TARGETS) ; do \
		make -C $$target $@ ; \
	done


