TOP := $(dir $(lastword $(MAKEFILE_LIST)))

default:
	more $(TOP)/README.txt

all: seq openmp cuda cuda_opt pgi hmpp

clean: 
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target clean ; \
	done

dist-clean: 
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target dist-clean ; \
	done


run_seq:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target run_seq ; \
	done

seq:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

run_openmp:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target run_openmp ; \
	done

openmp:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

run_cuda:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target run_cuda ; \
	done

cuda:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

run_cuda_opt:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target run_cuda_opt ; \
	done

cuda_opt:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

openmp_src:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

cuda_src:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

cuda_opt_src:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

pgi:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

run_pgi:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

hmpp:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

run_hmpp:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done


ppcg:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done
run_ppcg:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

