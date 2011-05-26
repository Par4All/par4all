/*Source Code From Laure Gonnord*/
//the famous car example

int undet();

int car(){
  int t,s,d;
  t=0;
  s=0;
  d=0;

 while(1){
   if (undet()) {
       while(t<=2){
	 s = 0;
	 t = t+1;
       }}
   if (undet()) {
    while (s<=1 && d<=8) {
      s = s+1;
      d = d+1;
    }
   }
 }
  
  return 0;
}

// loop invariant {d<=s+2t, s>=0, d>=s, s<=2, t<=3}
