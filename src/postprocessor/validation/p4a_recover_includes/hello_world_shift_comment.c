#include <stdio.h>/* Pour pouvoir utiliser proprement
   la fonction de sortie : */

#include <stdlib.h> /* Définit entre autres les codes de retour : */

/* To test avoiding '#include
   "/usr/lib/gcc/x86_64-linux-gnu/4.4.3/include/stdbool.h"' or '#include
   "/usr/lib/gcc/x86_64-linux-gnu/4.4.3/include-fixed/limits.h" */
#include <stdbool.h>
#include <limits.h>
#include <math.h>

int main(int argc, char *argv[]) {
  double d = HUGE_VAL;
  float f =  HUGE_VALF;
  long double ld = HUGE_VALL;
  /* Affiche sur la « sortie-standard » */
  puts("Hello, world!");
  // Renvoie une marque de réussite :
  return EXIT_SUCCESS;
}


