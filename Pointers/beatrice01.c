/* FI : case for shape analysis and descriptors and Feautrier's algorithm
 *
 * FI: I chagen the code to reduce its size.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX_PART 100
#define MAX_DOM 10
#define MIN_DIST 0.1

typedef struct part {
  float x, x_new;
  float y, y_new;
} part;

typedef struct dom {
  float x0, x_max;
  float y_0, y_max;
  part * tab_part[MAX_PART];
} dom;

typedef struct syst {
  float x_max, y_max;
  dom * tab_dom[MAX_DOM];
} syst;

float dist( part * p1, part *p2 ) {
  float dx = p1->x - p2->x;
  float dy = p1->y - p2->y;
  return ( sqrt( dx * dx + dy * dy ) );
}

int main()
{
  int i, j, k;
  float x, y;
  //syst * my_syst;
  //part * curr_part1, *curr_part2;

  // initializations
  syst * my_syst = (syst *) malloc( sizeof(syst) );
  for ( i = 0; i < MAX_DOM; i++ ) {
    my_syst->tab_dom[i] = (dom *) malloc( sizeof(dom) );
    for ( j = 0; j < MAX_PART; j++ )
      my_syst->tab_dom[i]->tab_part[j] = (part *) malloc( sizeof(part) );
  }
  x = 0; y = 0;

  // computations
  for ( i = 0; i < MAX_DOM; i++ ) {
    dom * curr_dom = my_syst->tab_dom[i];

    for ( j = 0; j < MAX_PART; j++ ) {
      part * curr_part1;
      curr_part1 = curr_dom->tab_part[j];

      curr_part1->x = x + 0.01 * (float) j;
      curr_part1->y = y + 0.01 * (float) j;
    }

    for ( j = 0; j < MAX_PART; j++ ) {
      part * curr_part1 = curr_dom->tab_part[j];

      for ( k = j + 1; k < MAX_PART; k++ ) {
	part * curr_part2 = curr_dom->tab_part[k];

	if ( dist( curr_part1, curr_part2 ) < MIN_DIST ) {
	  curr_part1->x_new = curr_part1->x - MIN_DIST;
	  curr_part1->y_new = curr_part1->y - MIN_DIST;
	}
      }
    }
  }

  // final IOs (& free()...)

  return 0;
}
