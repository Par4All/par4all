/*Source Code From Laure Gonnord*/
//Example coming from Goubault's and al paper in ESOP: 
// http://www.di.ens.fr/~goubault/GOUBAULTpapers.html

int esop_goubault2b(){
  
  int i,j;
  j=175;
  i=150;

  while(j>=100){
    if (j<=i-1) j=j-2;
    else ++i;
  }

  return 0;
}

//Invariant inside the while loop :   {j>=100, j<=175, i<=176}
// at the end : {i<=176, j<=99} 
