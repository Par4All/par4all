/** @file

    API of Par4All C to OpenMP

    This file is an implementation of the C to accelerator Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Péle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"
*/

#include <p4a_accel.h>
#include <errno.h>

/* target-independent parts */
#include "p4a_accel-shared.c"
/* target-dependent part. Each file must implement the p4a_accel.h interface not implemented in p4a_accel-shared.c */
#include "p4a_accel-cuda.c"
#include "p4a_accel-OpenCL.c"
#include "p4a_accel-OpenMP.c"
