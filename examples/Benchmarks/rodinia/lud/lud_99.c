/*
 * =====================================================================================
 *
 *       Filename:  suite.c
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

/*
 * Converted to C99 by Mehdi Amini (mehdi.amini@hpc-project.com) on 5th june 2011
 */


#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "timing.h"



typedef enum _FUNC_RETURN_CODE {
    RET_SUCCESS,
    RET_FAILURE
}func_ret_t;

func_ret_t create_matrix_from_file(int matric_dim, float m[matric_dim][matric_dim], const char* filename, int *size_p);
func_ret_t lud_verify(int matrix_dim, float m[matrix_dim][matrix_dim], float lu[matrix_dim][matrix_dim]);


static int do_verify = 0;

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"input", 1, NULL, 'i'},
      {"size", 1, NULL, 's'},
      {"verify", 0, NULL, 'v'},
      {0,0,0,0}
};


void lud_base(int size,float a[size][size])
{
     int i,j,k;

     // We cheat on the increment so that we won't parallelize this version 
     for (i=0; i<size; i++){
         for (j=i; j<size; ){
             float sum=a[i][j];
             for (k=0; k<i; ) {
               sum -= a[i][k]*a[k][j];
               k++;
             }
             a[i][j]=sum;
             j++;
         }

         for (j=i+1;j<size; ){
             float sum=a[j][i];
             for (k=0; k<i; ) {
               sum -=a[j][k]*a[k][i];
               k++;
             }
             a[j][i]=sum/a[i][i];
             j++;
         }
     }
}


// This is the reference to parallelize
void lud_99(int size, float a[size][size])
{
     int i,j,k;

     for (i=0; i<size; i++){
         for (j=i; j<size; j++){
             float sum=a[i][j];
             for (k=0; k<i; k++) sum -= a[i][k]*a[k][j];
             a[i][j]=sum;
         }

         for (j=i+1;j<size; j++){
             float sum=a[j][i];
             for (k=0; k<i; k++) sum -=a[j][k]*a[k][i];
             a[j][i]=sum/a[i][i];
         }
     }
}


int
main ( int argc, char *argv[] )
{
  int matrix_dim; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;

  while ((opt = getopt_long(argc, argv, "::vs:i:", 
                            long_options, &option_index)) != -1 ) {
      switch(opt){
        case 'i':
          input_file = optarg;
          break;
        case 'v':
          do_verify = 1;
          break;
        case 's':
          matrix_dim = atoi(optarg);
          break;
        case '?':
          fprintf(stderr, "invalid option\n");
          break;
        case ':':
          fprintf(stderr, "missing argument\n");
          break;
        default:
          fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
                  argv[0]);
          exit(EXIT_FAILURE);
      }
  }

  if ( (optind < argc) || (optind == 1)) {
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      exit(EXIT_FAILURE);
  }
  
  float m[matrix_dim][matrix_dim], mm[matrix_dim][matrix_dim];
  if (input_file) {
      ret = create_matrix_from_file(matrix_dim,m, input_file, &matrix_dim);
      ret = create_matrix_from_file(matrix_dim,mm, input_file, &matrix_dim);
      if (ret != RET_SUCCESS) {
          fprintf(stderr, "error create matrix from file %s\n", input_file);
          exit(EXIT_FAILURE);
      }
  } else {
    fprintf(stderr,"No input file specified!\n");
    exit(EXIT_FAILURE);
  } 

  
  /* Start timer. */
  timer_start();

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    memset(m,0,sizeof(mm));
  }

  
  lud_99(matrix_dim,mm);

  /* Cheat the compiler (again) to limit the scope of optimisation */
  if(argv[0]==0) {
    lud_verify(matrix_dim,mm, m);
  }

  /* Stop timer and display. */
  timer_stop_display();

  if (do_verify){
    lud_base(matrix_dim, m);
    fprintf(stderr,">>>Verify<<<<\n");
    lud_verify(matrix_dim,mm, m);
  }

  return EXIT_SUCCESS;
}				
/* ----------  end of function main  ---------- */


// Load input from file
func_ret_t create_matrix_from_file(int matric_dim, float m[matric_dim][matric_dim], const char* filename, int *size_p){
  int i, j, size;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if ( fp == NULL) {
      return RET_FAILURE;
  }

  fscanf(fp, "%d\n", &size);
  if(matric_dim!=size) {
    fprintf(stderr,"ERROR : -s incohérent avec le fichier d'input\n");
    exit(1);
  }

  if ( m == NULL) {
      fclose(fp);
      return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
          fscanf(fp, "%f ", &m[i][j]);
      }
  }

  fclose(fp);

  *size_p = size;

  return RET_SUCCESS;
}


// Verify results
func_ret_t lud_verify(int matrix_dim, float m[matrix_dim][matrix_dim], float lu[matrix_dim][matrix_dim]){
  int i,j;

  for (i=0; i<matrix_dim; i++){
      for (j=0; j<matrix_dim; j++){
          if ( fabs(m[i][j]-lu[i][j]) > 0.5)
            printf("dismatch at (%d, %d): (o)%f (n)%f\n", i, j, m[i][j], lu[i][j]);
      }
  }
  return 1;
}

