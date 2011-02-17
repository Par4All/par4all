#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MY_PI 3.14159265358979323846

typedef struct {
  float latitude;
  float longitude;
  float stock;
} town;

int main() {
  size_t i,j,k;
  float xmin, ymin, step, range;
  int rangex = 300;
  int rangey = 300;
  int nb = 3000;
  town pt[300][300], t[3000];

  xmin = ymin = step = range = 2.0;
  for (k=0; k<nb; k++) {
    t[k].latitude = k*step*180/MY_PI;
    t[k].longitude = k*step*180/MY_PI;
    t[k].stock = 500.;
  }

  for (i=0; i<rangex; i++)
    for (j=0; j<rangey; j++) {
      pt[i][j].latitude = (xmin+step*i)*180/MY_PI;
      pt[i][j].longitude = (ymin+step*j)*180/MY_PI;
      pt[i][j].stock = 0.;
      for (k=0; k<nb; k++) {
        float tmp = 6368.* acos(cos(xmin+step*i)*cos(t[k].latitude) *
                                cos((ymin+step*j)-t[k].longitude) +
                                sin(xmin+step*i)*sin(t[k].latitude));
        if (tmp < range)
          pt[i][j].stock += t[k].stock  / (1 + tmp);
      }
    }

  for (i=0; i<rangex; i++)
    for (j=0; j<rangey; j++)
      printf("%f %f %f\n", pt[i][j].latitude, pt[i][j].longitude, pt[i][j].stock);

  return 0;
}
