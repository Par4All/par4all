/* Test a mix of OpenMP pragma with comments and label.

   In OpenMP pragma must be between the label and the loop, if any.
*/

void pragma09()
{
  int i;

  // Comment
#pragma omp parallel for
  for(i = 0; i < 10; i++) {
    int j = i + 1;
  }
  // Comment before
#pragma omp parallel for
  // After
  for(i = 0; i < 10; i++) {
    int j = i + 1;
  }

 label1:
   #  \
 pragma omp parallel for
  for(i = 0; i < 10; i++) {
    int j = i + 1;
  }

 label2:
  /* Some comment */
   #  \
 pragma omp parallel for
   // And other
  for(i = 0; i < 10; i++) {
    int j = i + 1;
  }

   // Before
 label3:
  /* Some comment */
# pragma omp \
  parallel\
  for
    // And after
  for(i = 0; i < 10; i++) {
    int j = i + 1;
  }

  //Before
 far_away:
  //After
  ;
  // The end
}
