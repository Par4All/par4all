TOP := $(dir $(lastword $(MAKEFILE_LIST)))

default:
	more $(TOP)/USAGE

all: seq openmp cuda cuda_opt

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


