#include <stdlib.h>
int main(void){ int r=4096,k=128; double *x=malloc(sizeof(double)*r*k),*w=malloc(sizeof(double)*k),*e=malloc(sizeof(double)*r),*g=malloc(sizeof(double)*k);
 for(int i=0;i<r*k;i++)x[i]=0.5; for(int j=0;j<k;j++)w[j]=0.25; double total=0;
 for(int rep=0;rep<2000;rep++){ for(int i=0;i<r;i++){double s=0;for(int j=0;j<k;j++)s+=x[i*k+j]*w[j];e[i]=s-1;} for(int c=0;c<k;c++){double s=0;for(int q=0;q<r;q++)s+=e[q]*x[q*k+c];g[c]=s;} w[0]+=0.000000001*g[0]; total=w[0]; }
 return ((int)(total*1000.0))%100; }
