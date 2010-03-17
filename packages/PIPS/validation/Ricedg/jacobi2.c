//test de parallelisation sur la calcul du jacobien d'une image
//exemple qui marche, on leurre PIPS en créant un typedef

#include <stdlib.h>
#include <stdio.h>
typedef int* pint;

int jacobi2(int nb, pint* a, pint* b){
  int indice_i, indice_j;
  indice_i=0;
  indice_j=0;
  
  for( indice_i=1 ; indice_i<nb-1 ; indice_i++ ){
    for( indice_j=1 ; indice_j<nb-1 ; indice_j++){
      a[indice_i][indice_j] = ( 
			       b[indice_i-1][indice_j]
			       + b[indice_i+1][indice_j]
			       + b[indice_i][indice_j-1]
			       + b[indice_i][indice_j+1]
			       ) / 4 ;
    }
  }
  
 
  return 0;
}


int main(int argc, char *argv[]){
	//int i;for(i=0;i<argc;i++) printf("_%s_\n",argv[i]);
	pint* a;
	pint* b;
	int indice_i;
	if( argc > 1 ){
		int nb = atol(argv[1]);
		b = malloc(sizeof(pint)*(nb+1));
		a = malloc(sizeof(pint)*(nb+1));
	
		for( indice_i=0 ; indice_i<=nb ; indice_i++){
			a[indice_i] = malloc(sizeof(int)*(nb+1));
			b[indice_i] = malloc(sizeof(int)*(nb+1));
		}
  		return jacobi2(nb,a,b);
	}
	return 0;
}
