#include <stdio.h>
#include <stdlib.h>

#define NB_ELM 10



void init_tab_dynamique(double *table){
  int i;
  for (i=0; i < NB_ELM; i++) {
      table[i]= i*1.;
  }
}


void afficher(double tab[]){
  int i;
  for (i=0; i<NB_ELM; i++){
    printf("%f ", tab[i]);
  }
}

int main(){

  double maTable_statique[NB_ELM];

  double * maTable_dynamique = (double *) malloc(NB_ELM*sizeof(double));


  init_tab_dynamique(maTable_dynamique);
  afficher(maTable_statique);




  return 0;
}
