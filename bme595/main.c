#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
long double c_conv(int in_channel, long int o_channel, int kernel_size, int stride);




int main()
{
	int in_channel = 3;
	int o_channel = 1;
	int kernel_size = 3;
	int stride = 1;
	clock_t start;
	clock_t end;
	long double operations = 0.0;
	for(int i = 0; i < 11; i++){
		o_channel = pow(2,i);
		start = clock();
		operations = c_conv(in_channel, o_channel, kernel_size, stride);
		end = clock();
		printf("i = %d, o_channel = %d, number_of_operations = %Lf, computation_time = %lf \n", i, o_channel, operations, (double)(end - start) / CLOCKS_PER_SEC);

	}
   
   return 0;
}

long double c_conv(int in_channel, long int o_channel, int kernel_size, int stride){

//file handle
	FILE *myFile;
    myFile = fopen("testimage1.txt", "r");


//filesize
	int m = 720+2;
	int n = 1280+2;

//create image array with padding margins

	float **img_padded = (float**)malloc(m*sizeof(float*));
    for (int i = 0; i < m; i++) img_padded[i] = (float*)malloc(n*sizeof(float));
    
//read txt into img_padded
    for(int i = 0; i < m; i++){
    		for(int j = 0; j < n ; j++){
    			fscanf(myFile, "%f", &img_padded[i][j] );  

    		}
    }
    fclose(myFile);

//create output image array
	float ***img_output = (float***)malloc(o_channel*sizeof(float**));
    for (int i = 0; i < o_channel; i++)
        img_output[i] = (float**)malloc((m-2)*sizeof(float*));
    
    for (int i = 0; i < o_channel; i++) 
        for (int j = 0; j < (m-2); j++) 
            img_output[i][j] = (float*)malloc((n-2)*sizeof(float));

//generate kernel array

    float ***kernelArr = (float***)malloc(o_channel*sizeof(float**));
    for (int i = 0; i < o_channel; i++)
        kernelArr[i] = (float**)malloc(kernel_size*sizeof(float*));
    
    for (int i = 0; i < o_channel; i++) 
        for (int j = 0; j < kernel_size; j++) 
            kernelArr[i][j] = (float*)malloc(kernel_size*sizeof(float));

//generate random kernel value for convolution
    for(int i = 0; i < o_channel; i++)
        for(int j = 0; j < kernel_size; j++)
            for(int k = 0; k < kernel_size; k++){
                kernelArr[i][j][k] = (float)(rand()%20-10.0)/10.0;
            }


	long double operation_count = 0;

	for(int i = 1; i < m-1; i++){
		for(int j = 1; j < n-1; j++){

			//(i,j) pixel in the source image

			for(int q = 0; q < o_channel; q++){
				// use kernelArr[q] for convolution
				float** kernel = kernelArr[q];
				float tmpSum = 0.0;
				//loop the 3*3 window
				for(int k = 0; k < kernel_size; k++){
					for(int p =0; p < kernel_size; p++){
						tmpSum = tmpSum + kernel[k][p] * img_padded[i-1+k][j-1+p];
						operation_count = operation_count + 2;
					}
				}
				operation_count-- ;

				img_output[q][i-1][j-1] = tmpSum;

			}


		}


	}
	return operation_count;
}