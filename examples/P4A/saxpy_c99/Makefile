CCFLAGS = -Wall --std=c99

TARGETS =  saxpy-simple-seq saxpy-simple-cu saxpy-simple-omp saxpy-double-seq saxpy-double-cu saxpy-double-omp

all : $(TARGETS)

%-simple-cu : %.c
	p4a -C --c99 $< -o $@

%-simple-seq : %.c
	gcc $(CCFLAGS) $<  -o $@

%-simple-omp : %.c
	p4a -O $< -o $@

%-double-cu : %.c
	p4a -C --c99 -DUSE_DOUBLE $< -o $@

%-double-seq : %.c
	gcc $(CCFLAGS) -DUSE_DOUBLE $< -o $@

%-double-omp : %.c
	p4a -O -DUSE_DOUBLE $< -o $@

clean:
	@rm -rf $(TARGETS) *~ *.p4a.c
