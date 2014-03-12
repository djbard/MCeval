#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <unistd.h>



void checkCUDAerror(const char *msg);



// kernel to make the calculation. 
__global__ void calc(float* a1, float* b1, float* c1,float* a2, float* b2, float* c2, float* a3, float* b3, float* c3, float* a4, float* b4, float* c4, float* a5, float* b5, float* c5, float* a6, float* b6, float* c6, float* x, float* data, int ndata,  float* LH)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Need to calc the pearson chi2 : sum over data points of (model-data)^2/data
  // this will mean that x and data have the same range. 
  float chi2 = 0;
  for(int i=0;i<ndata;i++){
    // calc gaussian at this point
    float sumG=0;
    float xx = x[i];
    sumG += a1[idx]*expf( -1*(xx-b1[idx]*b1[idx])/(2*c1[idx]*c1[idx]) );
    sumG += a2[idx]*expf( -1*(xx-b2[idx]*b2[idx])/(2*c2[idx]*c2[idx]) );
    sumG += a3[idx]*expf( -1*(xx-b3[idx]*b3[idx])/(2*c3[idx]*c3[idx]) );
    sumG += a4[idx]*expf( -1*(xx-b4[idx]*b4[idx])/(2*c4[idx]*c4[idx]) );
    sumG += a5[idx]*expf( -1*(xx-b5[idx]*b5[idx])/(2*c5[idx]*c5[idx]) );
    sumG += a6[idx]*expf( -1*(xx-b6[idx]*b6[idx])/(2*c6[idx]*c6[idx]) );

    chi2+=( (data[i]-sumG)*(data[i]-sumG))/sumG;

  } 
  
  LH[idx] = chi2;



}



//////////////
int main (int argc, char **argv)
{

// do I have any input args? 
char* name;
if(argc>1)
  {
    name = argv[1];
  }


// how many walkers? They will be evaluated in parallel. 
int nWalkers = 80;


////////////////////////////
// now for the data. Assume it'll be a 30x30 pixel square.  
// I put this outside the loop cos the data never changes. 

int ndata = 30*30;
size_t sizeneeded_data = ndata*sizeof(float);

float *h_x = 0, *h_data = 0;
h_x = (float*) malloc(sizeneeded_data);
h_data = (float*) malloc(sizeneeded_data);
for(int i=0;i<ndata;i++){
h_x[i]=i;
h_data[i]=1;
}


// data GPU memory
float *d_x, *d_data;
cudaMalloc(&d_x, sizeneeded_data);
cudaMalloc(&d_data, sizeneeded_data);


cudaMemcpy(d_data, h_data, sizeneeded_data, cudaMemcpyHostToDevice);
cudaMemcpy(d_x, h_x, sizeneeded_data, cudaMemcpyHostToDevice);


checkCUDAerror("data memcpy");
//////////////////test////////////////////
// try looping over 1000 times. I want to see how long this takes, in this dumb implementation. 
////////////////////////////////////////
for(int hh=0;hh<1000;hh++){

// set up the walkers CPU memory. 
size_t sizeneeded = nWalkers*sizeof(float);
float *h_a1 = 0, *h_b1=0, *h_c1=0;
float *h_a2 = 0, *h_b2=0, *h_c2=0;
float *h_a3 = 0, *h_b3=0, *h_c3=0;
float *h_a4 = 0, *h_b4=0, *h_c4=0;
float *h_a5 = 0, *h_b5=0, *h_c5=0;
float *h_a6 = 0, *h_b6=0, *h_c6=0;

h_a1 = (float*) malloc(sizeneeded);
h_b1 = (float*) malloc(sizeneeded);
h_c1 = (float*) malloc(sizeneeded);
h_a2 = (float*) malloc(sizeneeded);
h_b2 = (float*) malloc(sizeneeded);
h_c2 = (float*) malloc(sizeneeded);
h_a3 = (float*) malloc(sizeneeded);
h_b3 = (float*) malloc(sizeneeded);
h_c3 = (float*) malloc(sizeneeded);
h_a4 = (float*) malloc(sizeneeded);
h_b4 = (float*) malloc(sizeneeded);
h_c4 = (float*) malloc(sizeneeded);
h_a5 = (float*) malloc(sizeneeded);
h_b5 = (float*) malloc(sizeneeded);
h_c5 = (float*) malloc(sizeneeded);
h_a6 = (float*) malloc(sizeneeded);
h_b6 = (float*) malloc(sizeneeded);
h_c6 = (float*) malloc(sizeneeded);

// assign them random numbers. 
for(int i=0;i<nWalkers;i++){
  h_a1[i]=0.1;
  h_b1[i]=0.5;
  h_c1[i]=10.0;
  h_a2[i]=0.1;
  h_b2[i]=0.5;
  h_c2[i]=10.0;
  h_a3[i]=0.1;
  h_b3[i]=0.5;
  h_c3[i]=10.0;
  h_a4[i]=0.1;
  h_b4[i]=0.5;
  h_c4[i]=10.0;
  h_a5[i]=0.1;
  h_b5[i]=0.5;
  h_c5[i]=10.0;
  h_a6[i]=0.1;
  h_b6[i]=0.5;
  h_c6[i]=10.0;
}

// assign the GPU memory
float *d_a1, *d_b1, *d_c1;
float *d_a2, *d_b2, *d_c2;
float *d_a3, *d_b3, *d_c3;
float *d_a4, *d_b4, *d_c4;
float *d_a5, *d_b5, *d_c5;
float *d_a6, *d_b6, *d_c6;
cudaMalloc(&d_a1, sizeneeded);
cudaMalloc(&d_b1, sizeneeded);
cudaMalloc(&d_c1, sizeneeded);
cudaMalloc(&d_a2, sizeneeded);
cudaMalloc(&d_b2, sizeneeded);
cudaMalloc(&d_c2, sizeneeded);
cudaMalloc(&d_a3, sizeneeded);
cudaMalloc(&d_b3, sizeneeded);
cudaMalloc(&d_c3, sizeneeded);
cudaMalloc(&d_a4, sizeneeded);
cudaMalloc(&d_b4, sizeneeded);
cudaMalloc(&d_c4, sizeneeded);
cudaMalloc(&d_a5, sizeneeded);
cudaMalloc(&d_b5, sizeneeded);
cudaMalloc(&d_c5, sizeneeded);
cudaMalloc(&d_a6, sizeneeded);
cudaMalloc(&d_b6, sizeneeded);
cudaMalloc(&d_c6, sizeneeded);




///////////////////////////////
// assign the output memory. One number returned for each walker. 
size_t sizeneeded_out = nWalkers*sizeof(float);
float *h_LH = 0; 
float *d_LH;
h_LH = (float*) malloc(sizeneeded_out);
cudaMalloc(&d_LH, sizeneeded_out);


/////////////////////////
// copy data over to GPU
cudaMemcpy(d_a1, h_a1, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_b1, h_b1, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_c1, h_c1, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_a2, h_a2, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_b2, h_b2, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_c2, h_c2, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_a3, h_a3, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_b3, h_b3, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_c3, h_c3, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_a4, h_a4, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_b4, h_b4, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_c4, h_c4, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_a5, h_a5, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_b5, h_b5, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_c5, h_c5, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_a6, h_a6, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_b6, h_b6, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_c6, h_c6, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_LH, h_LH, sizeneeded_out, cudaMemcpyHostToDevice);

checkCUDAerror("memcpy");
   


// set up kernel params. 
// First: 80 walkers, each will eval one gaussian. 
int threadsPerBlock = 512; // max possible. Don't care much about mem access yet. 
int blocksPerGrid = int(ceil(nWalkers / float(threadsPerBlock)));
    //printf(" theads per block: %d and blocks per grid: %d for a total of: %d\n", threadsPerBlock, blocksPerGrid, threadsPerBlock*blocksPerGrid);


// run it! 
calc<<<blocksPerGrid, threadsPerBlock >>> (d_a1, d_b1, d_c1, d_a2, d_b2, d_c2, d_a3, d_b3, d_c3, d_a4, d_b4, d_c4, d_a5, d_b5, d_c5, d_a6, d_b6, d_c6, d_x, d_data, ndata, d_LH);

checkCUDAerror("kernel");
    


// copy the data back off the GPU
cudaMemcpy(h_LH, d_LH, sizeneeded_out, cudaMemcpyDeviceToHost);

// print it out...
if(hh==500){
for(int i=0;i<nWalkers;i++){
  printf("LH is: %f  ", h_LH[i]);
}
}


//printf("\n");

//////////////
// now free upp all the histo memory, both on CPU and GPU. Otherwise I'll fill up the device memory pretty fast! 
free(h_a1); 
free(h_b1);
free(h_c1);
free(h_a2);
free(h_b2);
free(h_c2);
free(h_a3); 
free(h_b3);
free(h_c3);
free(h_a4);
free(h_b4);
free(h_c4);
free(h_a5);
free(h_b5);
free(h_c5);
free(h_a6);
free(h_b6);
free(h_c6);
cudaFree(d_a1);
cudaFree(d_b1);
cudaFree(d_c1);
cudaFree(d_a2);
cudaFree(d_b2);
cudaFree(d_c2);
cudaFree(d_a3);
cudaFree(d_b3);
cudaFree(d_c3);
cudaFree(d_a4);
cudaFree(d_b4);
cudaFree(d_c4);
cudaFree(d_a5);
cudaFree(d_b5);
cudaFree(d_c5);
cudaFree(d_a6);
cudaFree(d_b6);
cudaFree(d_c6);


}// end loop over 1000 walker updates. 


free(h_x);
free(h_data);
cudaFree(d_x);
cudaFree(d_data);
}


//////////////////////////////////////////////////////////////
//  simple function to check for errors. 
//////////////////////////////////////////////////////////////

void checkCUDAerror(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
              cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}


