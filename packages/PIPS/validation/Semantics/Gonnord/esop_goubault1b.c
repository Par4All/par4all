/*Source Code From Laure Gonnord*/
//Example coming from Goubault's and al paper in ESOP: 
// http://www.di.ens.fr/~goubault/GOUBAULTpapers.html

int esop_goubault1b(){
  
  int i,j;
  j=10;
  i=1;

  i=i+2;
  j=j-1;

  if (random()) {i=i+2;j=j-1;}
  else
    {i=i+2;j=j-1;}

  while(i<=j) {i=i+2;j=j-1;}

  return 0;
}

//Invariant inside the while loop :    = {i>=5, i<=9, i+2j=21}
//at the end of the program : {3i>=23, i<=9, i+2j=21}
