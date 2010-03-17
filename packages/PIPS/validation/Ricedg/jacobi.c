//test de parallelisation sur la calcul du jacobien d'une image
//exemple ne marchant pas

#include <stdlib.h>
#include <stdio.h>

int jacobi(int nb, int** a, int** b){
  int indice_i, indice_j;
  
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
	int** a;
	int** b;
	int indice_i;
	if( argc > 1 ){
		int nb = atol(argv[1]);
		b = malloc(sizeof(int*)*(nb+1));
		a = malloc(sizeof(int*)*(nb+1));
	
		for( indice_i=0 ; indice_i<=nb ; indice_i++){
			a[indice_i] = malloc(sizeof(int)*(nb+1));
			b[indice_i] = malloc(sizeof(int)*(nb+1));
		}
  		return jacobi(nb,a,b);
	}
	return 0;
}
