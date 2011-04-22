//     goal: show effect of cloning, partial evaluation and loop unrolling
//     and reduction parallelization for a Power architecture
//     kernel_size must be odd
#include <stdio.h>
#include "tools.h"
#define kernel_size 3
#define nsteps 10

void convol3x3();

int main(int argc, char **argv) //      program image_processing
{
	int image_size;
	float (*image)[image_size][image_size];
	float (*new_image)[image_size][image_size];
	float kernel[kernel_size][kernel_size];
	int i, j, n;

	if (argc < 3)
	{
		fprintf(stderr, "Usage: %s image_size data_file\n", argv[0]);
		return 1;
	}
	image_size = atoi(argv[1]);
	image = malloc(sizeof(float)*image_size*image_size);
	new_image = malloc(sizeof(float)*image_size*image_size);
	init_data_file(argv[2]);
	init_data_float(image, image_size*image_size);
	init_data_float(new_image, image_size*image_size);
	close_data_file();


	for( i = 0; i< kernel_size; i++) {
		for( j = 0; j< kernel_size; j++) {
			kernel[i][j] = 1;
		}
	}

	//     read *, image
	for( i = 0; i< image_size; i++) {
		for( j = 0; j< image_size; j++)
			(*image)[i][j] = 1.;
	}


	for( n = 0; n< nsteps; n++) {
		convol3x3(image_size, image_size, *new_image, *image,
				kernel);
	}

//	for( i = 0; i< image_size; i++) { 
//		printf("%f ",(*new_image)[i][i]); 
//	} 
	free(image);
	free(new_image);
	//     print *, new_image
	//      print *, new_image (image_size/2, image_size/2)
	print_array_float("new_image", new_image, image_size*image_size);

	return 0; 
}

void convol3x3(int isi, int isj, float new_image[isi][isj], float image[isi][isj], 
		float kernel[kernel_size][kernel_size])
{
	//     The convolution kernel is not applied on the outer part
	//     of the image

	int i, j;

	for(i = 0; i< isi; i++) {
		for(j = 0; j< isj; j++) {
			new_image[i][j] = image[i][j];
		}
	}

	for(i =  kernel_size/2; i<isi - kernel_size/2; i++) {
		for(j =  kernel_size/2; j<isj - kernel_size/2; j++) {
			new_image[i][j] = 0.;
			new_image[i][j] += image[i+0-kernel_size/2][j+0-kernel_size/2]* kernel[0][0];
			new_image[i][j] += image[i+0-kernel_size/2][j+1-kernel_size/2]* kernel[0][1];
			new_image[i][j] += image[i+0-kernel_size/2][j+2-kernel_size/2]* kernel[0][2];
			new_image[i][j] += image[i+1-kernel_size/2][j+0-kernel_size/2]* kernel[1][0];
			new_image[i][j] += image[i+1-kernel_size/2][j+1-kernel_size/2]* kernel[1][1];
			new_image[i][j] += image[i+1-kernel_size/2][j+2-kernel_size/2]* kernel[1][2];
			new_image[i][j] += image[i+2-kernel_size/2][j+0-kernel_size/2]* kernel[2][0];
			new_image[i][j] += image[i+2-kernel_size/2][j+1-kernel_size/2]* kernel[2][1];
			new_image[i][j] += image[i+2-kernel_size/2][j+2-kernel_size/2]* kernel[2][2];
			new_image[i][j] = new_image[i][j]/(kernel_size*kernel_size);
			//new_image[i][j] = new_image[i][j]*0.111111;
		}
	}
}
