//     Check nest parallelization: the two loops are parallel, but they
//    must also be interchanged

// Make sure that the longuest loop direction is not chosen when
// contiguity does impact the decision

void nest04()
{
  float a[10][20];
  float b[20][10];
  int i, j;

  for(j = 0; j<20; j++)
    for(i = 0; i<10; i++)
      a[i][j] = b[j][i]+a[i][j];

}

