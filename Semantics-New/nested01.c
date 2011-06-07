// Make sure that internal loops are properly processed when they are
// strictly nested

// FI: No problem if for loops are used, although I do not understand
// why when looking at PIPS source code for... while loops.

// FI: tortured while loops like these end up with a wrong semantic analysis

#include <assert.h>

main()
{
  int i=0, j, m, n, k =0;

  //assert(m>=1 && n>=1);

  // Try first something simpler than a while loop
  if(i++, j=0, i<m) {
    k = 0;
  }

  while(i++, j=0, i<m)
    while(j++, j<n)
      k++;

  // Check the loop nest postcondition
  i = i;
}
