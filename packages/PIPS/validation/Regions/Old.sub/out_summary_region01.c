
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"input", 1, NULL, 'i'},
      {"size", 1, NULL, 's'},
      {"verify", 0, NULL, 'v'},
      {0,0,0,0}
};

// out_summary_regions musn't core dump with argv declared as an array
int my_getopt_long(int argc, char * const argv[],
                const char *optstring,
                const struct option *longopts, int *longindex) {
  return rand()-1;
}

int
main ( int argc, char *argv[] )
{
  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  const char *input_file = NULL;
  float *m, *mm;
  int do_verify;
  while ((opt = my_getopt_long(argc, argv, "::vs:i:",
                            long_options, &option_index)) != -1 ) {
      switch(opt){
        case 'v':
          break;
        default:
          do_verify = 1;
      }
  }

  if (do_verify){
    //printf("After LUD\n");
    //print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
  }

  return EXIT_SUCCESS;
}

