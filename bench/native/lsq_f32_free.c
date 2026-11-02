#include <stdlib.h>
int main(void){ int r=4096,k=128; float *x=malloc(sizeof(float)*r*k),*w=malloc(sizeof(float)*k),*e=malloc(sizeof(float)*r),*g=malloc(sizeof(float)*k);
 for(int i=0;i<r*k;i++)x[i]=0.5; for(int j=0;j<k;j++)w[j]=0.25; float total=0;
 for(int rep=0;rep<2000;rep++){ for(int i=0;i<r;i++){float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;for(int j=0;j<k;j+=8){s0+=x[i*k+j]*w[j];s1+=x[i*k+j+1]*w[j+1];s2+=x[i*k+j+2]*w[j+2];s3+=x[i*k+j+3]*w[j+3];s4+=x[i*k+j+4]*w[j+4];s5+=x[i*k+j+5]*w[j+5];s6+=x[i*k+j+6]*w[j+6];s7+=x[i*k+j+7]*w[j+7];}e[i]=((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7))-1;}
 for(int c=0;c<k;c++)g[c]=0; for(int q=0;q<r;q++){float eq=e[q];for(int c=0;c<k;c++)g[c]+=eq*x[q*k+c];} w[0]+=0.000000001f*g[0]; total=w[0]; }
 return ((int)(total*1000.0f))%100; }
