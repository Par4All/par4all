If you do not have write access to this directory, copy its content (with
cp -a for example) somewhere you can write into and try the examples from
this new location.

# To make the demo
make demo

This small examples illustrates the code generation for the SCMP
architecture using the Sesam HAL.

The source file (data_flow01.c) contains an external loop which chains
two tasks identifed by labels (scmp_task_0 and scmp_task_1).

The p4a scmp compiler generates code for an independent process per
task (which is called a kernel task) and for an independent process
per additional task in charge of ensuring the consistency of data
(this is called a server task).

The "make demo" command automatically generates two directories:

- applis_processing/project_name contains:

  - the code for the kernel tasks (T001.mips.c and T002.mips.c);
    their only difference is their return value which is a
    synchronisation event value defined in a header file generated
    at compile time (*_event_val.h), and a #define at the beginning
    which is used to specialize the code (see the if scmp_task_0_p
    a littel further in the code for instance);

  - the code for the (here unique) server task (T003.mips.c); the
    differences with T001.mips.c are here again the return value and
    the #define at the beginning, but also the names of the P4A
    run-time functions which are specific versions for server tasks;

  - the buffer description header file (scmp_buffers.h), which defines
    constants used to further specialize the tasks depending on their
    communication buffers usage;

  - the Makefile.arp file is used to compile and build the whole
    application to run on the Sesam simulator;

- the applis/ directory contains a *.app file which is the code
  for the control task.
