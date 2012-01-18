# A GNU makefile to run a demo of Par4All on Jacobi application excerpt
LOCAL_TARGET= jacobi

SOURCES= $(LOCAL_TARGET:=.c)

RUN_ARG=Logo_HPC-Project-GTC.pgm

CLEAN_OTHERS += output.pgm output_accel-openmp.pgm output_cuda-opt.pgm \
	output_cuda.pgm output_opencl.pgm output_openmp.pgm output_seq.pgm


display% : run%
	mv output.pgm output$*.pgm
	# Display graphically the results:
	@echo "Please close the window at the end of the execution"
	eog output$*.pgm

#display_cuda-opt:
#	@echo "CUDA optimisation is not yet supported"

check% : run%
	mv output.pgm output$*.pgm
	# Compare the result with the reference. Note that there may be
	# some slight differences because of non associativity of floating
	# point computations or different implementations of floating
	# point computations on GPUs:
	-diff -q output$*.pgm output-ref.pgm

