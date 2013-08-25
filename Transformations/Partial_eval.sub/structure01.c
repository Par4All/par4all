
#define dim 3
#define dim1 dim
#define dim2 2*dim

typedef struct{
  int re;
  int im;
  int index;
} TNUMBER;

typedef struct{
  int re;
  int im;
} NUMBER;


// void run(int d1, int d2,
//          TNUMBER  *inputin)
// {
//   NUMBER value[d1][d2];
//   int i, k;
//   TNUMBER input;
//   
//   for(i=0; i<d1; i++)
//   { 
//     input= inputin[i];
//     k = input.index;
//     if ((k>=0 ) && (k<d2))
//     {
//       value[i][k].re = input.re;
//       value[i][k].im = input.im;
//     }
//   }
// 
// }


int main()
{
  int i, k, a[dim];
//   int dim=2;
  TNUMBER structure[dim];
  
  loop1:
  for (i=0; i<dim; i++)
  {
    structure[i].index = 2*i;
    structure[i].re = i;
    structure[i].im = i*i;
  }
  
  return 0;
}


  
//   
//   NUMBER * value[dim1][dim2];
//   TNUMBER temp;
//   
//   loop2:
//   for(i=0; i<dim; i++)
//   { 
//     temp= structure[i];
//     k = temp.index;
//     if ((k>=0) && (k<dim2))
//     {
//       value[i][k]->re = temp.re;
//       value[i][k]->im = temp.im;
//     }
//   }



