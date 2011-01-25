#ifndef __PIPS__
#define MIN(a,b) ((a)<(b))?(a):(b)
#endif
#include <stdio.h>
#define kernel_size 5

void erode(int isi, int isj, int new_image[isi][isj], int image[isi][isj])
{

	int i, j, k,l;
	for(l=0;l<5;l++)

	for(i =  kernel_size/2; i<isi - kernel_size/2; i++) {
		for(j =  kernel_size/2; j<isj - kernel_size/2; j++) {
			int l=image[i][j];
			for(k=0;k<kernel_size;k++) {
				l = MIN(l,image[i][j+1-kernel_size/2+k]);
			}
			new_image[i][j] = l;
		}
	}
}

int main(int argc, char **argv) //      program image_processing
{
    int image_size = atoi(argv[1]);
    if(image_size > kernel_size ) {
        int (*image)[image_size][image_size];
        int (*new_image)[image_size][image_size];
        int i, j, k, n;
        image = (int(*)[image_size][image_size])malloc(sizeof(int)*image_size*image_size);
        new_image =(int(*)[image_size][image_size]) malloc(sizeof(int)*image_size*image_size);
        for( i = 0; i< image_size; i++)
            for( j = 0; j< image_size; j++) 
                (*new_image)[i][j]=i*j; 

        for(k=0;k<3;k++) {
            erode(image_size, image_size, *new_image, *image);
            erode(image_size, image_size, *image, *new_image);
        }

        for( i = 0; i< image_size; i++) { 
            printf("%f ",(*image)[i][i]); 
        } 
        free(image);
        free(new_image);
    }

    return 0; 
}

