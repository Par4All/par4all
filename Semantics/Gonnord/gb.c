/*Source Code From Laure Gonnord*/
// The famous gas burner !

int boolundet();

void gb(){
  int x,l,t;
  int bleak,bchange;
  
  x=0;l=0;t=0;
  bleak = 1;

  while(1){
    if(bleak == 1) // leaking
      {
	if (x>=10 || random()) {
	  x = 0;
	  bleak = 0;
	} else {
	  l=l+1;
	  x=x+1;
	  t=t+1;
	}
      }
    else //non leaking
      {
	if (x>=50 && random()) {
	  x = 0;
	  bleak = 1;
	} else {
	  x=x+1;
	  t=t+1;
	}
      }
  }
}

//invariant for (bleak==0) {l>=0, x>=0, l+x<=t, 6l+x<=t+50, bleak=0} 
// for (bleak == 1) {l>=0, t>=x, x>=0, 6l<=t+5x, 6l+x<=t+60, bleak=1}
//aspic -delay 65 :-( (accel for circuits not yet implemented)
