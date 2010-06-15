/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 *
 * Sample MPI "hello world" application in C
 */

#include <unistd.h>
#include <stdio.h>
#include "mpi.h"

#define LEN 80

int main(int argc, char* argv[])
{
  char nom[LEN];
  int rank, size;
  
  gethostname(nom, LEN);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("%s : Hello, world, I am process %d of %d\n", nom, rank, size);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}
