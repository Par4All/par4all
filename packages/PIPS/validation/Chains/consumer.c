typedef struct {
  float   re;
  float   im;
} Cplfloat;


// If produce is declare in the same compilation unit as consumer, there's no
// issue. The typedef is probably the source of the issue here.
void producer(Cplfloat array1[10]);

void consumer(Cplfloat array1[10],Cplfloat array2[10]) {
  array2[0].re=array1[0].re;
}


void producer_consumer( ){
  Cplfloat array1[10];
  Cplfloat array2[10];

  // We expect a dependence between producer and consumer
  producer(array1); 
  consumer(array1, array2);
  return;
}


