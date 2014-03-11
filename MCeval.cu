#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <assert.h>

#include <unistd.h>




// kernel to make the calculation. 
__global__ void calc(float* a, float* b, float* c, float* x, float* data, int ndata,  float* LH)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Need to calc the pearson chi2 : sum over data points of (model-data)^2/data
  // this will mean that x and data have the same range. 
  float chi2 = 0.0, thisG = 0.0;
  for(int i=0;i<ndata;i++){
    // calc gaussian at this point
    thisG = a[idx]*expf( -1*(x[i]-b[idx]*b[idx])/(2*c[idx]*c[idx]) );

    chi2+=( (data[i]-thisG)*(data[i]-thisG))/thisG;

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


// set up the walkers CPU memory. 
size_t sizeneeded = nWalkers*sizeof(float);
float *h_a = 0;
float *h_b = 0;
float *h_c = 0;

h_a = (float*) malloc(sizeneeded);
h_b = (float*) malloc(sizeneeded);
h_c = (float*) malloc(sizeneeded);

// assign them random numbers. 
for(int i=0;i<nWalkers;i++){
  h_a[i]=0.1;
  h_b[i]=0.5;
  h_c[i]=10.0;
}

// assign the GPU memory
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, sizeneeded);
cudaMalloc(&d_b, sizeneeded);
cudaMalloc(&d_c, sizeneeded);


// now for the data. Assume it'll be a 30x30 pixel square.  
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


// assign the output memory. One number returend for each walker. 
size_t sizeneeded_out = nWalkers*sizeof(float);
float *h_LH = 0; 
float *d_LH;
h_LH = (float*) malloc(sizeneeded_out);
cudaMalloc(&d_LH, sizeneeded_out);


//. copy data over to GPU
cudaMemcpy(d_a, h_a, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_c, h_c, sizeneeded, cudaMemcpyHostToDevice);
cudaMemcpy(d_x, h_x, sizeneeded_data, cudaMemcpyHostToDevice);
cudaMemcpy(d_data, h_data, sizeneeded_data, cudaMemcpyHostToDevice);
cudaMemcpy(d_LH, h_LH, sizeneeded_out, cudaMemcpyHostToDevice);


// set up kernel params. 
// First: 80 walkers, each will eval one gaussian. 
int threadsPerBlock = 1024; // max possible. Don't care much about mem access yet. 
int blocksPerGrid = int(ceil(nWalkers / float(threadsPerBlock)));
    printf(" theads per block: %d and blocks per grid: %d for a total of: %d\n", threadsPerBlock, blocksPerGrid, threadsPerBlock*blocksPerGrid);


// run it! 
calc<<<blocksPerGrid, threadsPerBlock >>> (d_a, d_b, d_c, d_x, d_data, ndata, d_LH);


// copy the data back off the GPU
cudaMemcpy(h_LH, d_LH, sizeneeded_out, cudaMemcpyDeviceToHost);

// print it out...
for(int i=0;i<nWalkers;i++){
  printf("LH is: %f  ", h_LH[i]);
}

printf("\n");

}