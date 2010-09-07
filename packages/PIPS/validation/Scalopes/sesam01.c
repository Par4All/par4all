#define SIZE 100

//global variables
int data1[SIZE];
int data2[SIZE];

main(){
    int i;
    for(i=0;i<SIZE;i++){
        data1[i]=i;
    }

    for(i=0;i<SIZE;i++){
        data2[i] = data1[i]*2+1;
    }
}
