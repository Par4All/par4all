#include <stdio.h>
void convol();
#define kernel_size 3
#define image_size 512

int main() 
{
  const int nsteps=20;
  short image[image_size][image_size];
  short new_image[image_size][image_size];
  short kernel[kernel_size][kernel_size];

  int i, j, n;

  for( i = 0; i< kernel_size; i++) {
    for( j = 0; j< kernel_size; j++) {
      kernel[i][j] = 1;
    }
  }

  for( i = 0; i< image_size; i++) {
    for( j = 0; j< image_size; j++)
      image[i][j] = 1.;
  }


  for( n = 0; n< nsteps; n++) {
    convol(new_image, image, kernel);
  }

  for( i = 0; i< image_size; i++) {
    for( j = 0; j< image_size; j++)
      printf("%f ",new_image[i][j]);
  }

  return 1; 
}

void convol(short new_image[image_size][image_size], short image[image_size][image_size],
    short kernel[kernel_size][kernel_size])
{
  //     The convolution kernel is not applied on the outer part
  //     of the image
  int i, j;
  for(i = 0; i< image_size; i++)
    for(j = 0; j< image_size; j++)
      new_image[i][j] = image[i][j];

here:  for(i =  kernel_size/2; i<image_size - kernel_size/2; i++)
    for(j =  kernel_size/2; j<image_size - kernel_size/2; j++)
      run_kernel(i,j,new_image,image,kernel);
}

void run_kernel(int i, int j, short new_image[image_size][image_size], short image[image_size][image_size], short kernel[kernel_size][kernel_size])
{
  int ki,kj;
  new_image[i][j] = 0;
  for(ki = 0; ki<kernel_size; ki++)
    for(kj = 0; kj<kernel_size; kj++)
      new_image[i][j] = new_image[i][j] + 
        image[i+ki-kernel_size/2][j+kj-kernel_size/2]* 
        kernel[ki][kj];
  new_image[i][j] = new_image[i][j]/(kernel_size*kernel_size);
}
