#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <tuple>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <bits/stdc++.h>
#include <cuda_runtime.h>

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

using namespace std;

float distanceSquared(ArffInstance* a, ArffInstance* b){
    float sum = 0;
    for (int i = 0; i < a->size()-1; i++){
        float diff = (a->get(i)->operator float() - b->get(i)->operator float());
        sum += diff*diff;
    }
    return sum;
}

__global__ void pairwiseDistanceKernel(float* X_data, float* distances, int num_instances, int num_features){
	//computes pairwise distances between instances in X_data
	//each thread computes the distance between instance i and instance j.
	int instance_i = blockDim.x * blockIdx.x + threadIdx.x;
	int instance_j = blockDim.y * blockIdx.y + threadIdx.y;

	//skip threads on the edge blocks that cross the matrix boundary
	if (instance_i >= num_instances || instance_j >= num_instances){return;}
	if (instance_i < instance_i){return;}

	float squared_distance = 0;
	for(int f = 0; f < num_features; f++){
		float difference = X_data[instance_i*num_features + f] - X_data[instance_j*num_features + f];
		squared_distance += difference*difference;
	}

	distances[num_instances*instance_j + instance_i] = distances[num_instances*instance_i + instance_j]  = squared_distance;

}

__global__ void perInstanceKNN_LinearReduction(float* distances, float* classes, int k, int num_instances){
	//parallelizes along the data dimension a minimum reduction
	//selection sort for smallest k
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid > num_instances) return;
//	printf("%i\n", tid);
	distances[tid*num_instances + tid] = FLT_MAX;
	for(int neighbor = 0; neighbor < k; neighbor++){
		int min_index = neighbor;
		for(int i = neighbor+1; i < num_instances; i++){
			if(distances[tid*num_instances + i] < distances[tid*num_instances + min_index]){
				min_index = i;
			}
		}

		//swap
		float temp = classes[tid*num_instances + neighbor];
		classes[tid*num_instances + neighbor] = classes[tid*num_instances + min_index];
		classes[tid*num_instances + min_index] = temp;

		temp = distances[tid*num_instances + neighbor];
		distances[tid*num_instances + neighbor] = distances[tid*num_instances + min_index];
		distances[tid*num_instances + min_index] = temp;

	}

}

int* KNN(ArffData* dataset, int k){
    //Implements a sequential kNN where for each candidate query an in-place priority queue
    //is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the dataset instances
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));

    //stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float* candidates = (float*) calloc(k*2, sizeof(float));
    for(int i = 0; i < 2*k; i++){candidates[i] = FLT_MAX;}

    //Compute number of classes
    int NUM_CLASSES = dataset->num_classes();
//    for(int i = 0; i < dataset->num_instances(); i++){
//        int class_index = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator long();
//        if(class_index+1 > NUM_CLASSES){
//            NUM_CLASSES = class_index+1;
//        }
//    }

    //stores bincounts of each class over the final set of candidate NN
    int* classCounts = (int*)calloc(NUM_CLASSES, sizeof(int));


    for(int queryIndex = 0; queryIndex < dataset->num_instances(); queryIndex++){
        for(int keyIndex = 0; keyIndex < dataset->num_instances(); keyIndex++){
            if (queryIndex == keyIndex) continue;
            float d_squared = distanceSquared(dataset->get_instance(queryIndex), dataset->get_instance(keyIndex));

            // add to our candidates
            for(int c = 0; c < k; c++){
                if(d_squared < candidates[2*c]){
                    //Found a new candidate
                    //Shift previous candidates down by one
                    for(int x = k-2; x >= c; x--){
                        candidates[2*x+2] = candidates[2*x];
                        candidates[2*x+3] = candidates[2*x+1];
                    }
                    //set key vector as potential k NN
                    candidates[2*c] = d_squared;
                    candidates[2*c+1] = dataset->get_instance(keyIndex)->get(dataset->num_attributes() - 1)->operator float();

                    break;
                }
            }
        }

        //bincount the candidate labels and pick the most common
        for(int i = 0; i < k;i++){
            classCounts[(int)candidates[2*i+1]] += 1;
        }
        int max = -1;
        int max_index = 0;
        for(int i = 0; i < NUM_CLASSES;i++){
            if(classCounts[i] > max){
                max = classCounts[i];
                max_index = i;
            }
        }

        predictions[queryIndex] = max_index;

        for(int i = 0; i < 2*k; i++){candidates[i] = FLT_MAX;}
        memset(classCounts, 0, NUM_CLASSES * sizeof(int));
    }
    
    return predictions;
}

int* KNN_GPU(ArffData* dataset, int k){
	//
	// Performs a (num_instances * num_feature) x (num_feature * num_instances)^T tiled matrix difference and squaring
	// This generates a pairwise distance from every instance to every other instance
	// The k nearest neighbors are then identified with a k-nary min reduction over each instance.
	//

	// predictions is the array where you have to return the class predicted (integer) for the dataset instances
	int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
	int NUM_CLASSES = dataset->num_classes();

	int num_features = dataset->num_attributes()-1;
	int num_elements = dataset->num_instances() * num_features;

	float* h_X_data  = (float*)malloc( num_elements * sizeof(float) );
	float* h_pairwise_distances  = (float*)malloc( dataset->num_instances()*dataset->num_instances() * sizeof(float) );
	float* h_classes  = (float*)malloc( dataset->num_instances()*dataset->num_instances() * sizeof(float) );

	//loads data from ARFF format in flat array
	for(int instance = 0; instance < dataset->num_instances(); instance++){
		for(int instance2 = 0; instance2 < dataset->num_instances(); instance2++){
			h_classes[instance*dataset->num_instances() + instance2] = dataset->get_instance(instance2)->get(num_features)->operator float();
		}
		for(int feature = 0; feature < num_features; feature++){
			h_X_data[instance*num_features + feature] = dataset->get_instance(instance)->get(feature)->operator float();
		}
	}

	float *d_X_data, *d_pairwise_distances, *d_classes;

	cudaMalloc(&d_X_data, num_elements * sizeof(float));
	cudaMalloc(&d_pairwise_distances, dataset->num_instances() * dataset->num_instances() * sizeof(float));
	cudaMalloc(&d_classes, dataset->num_instances() * dataset->num_instances() * sizeof(float));

	cudaMemcpy(d_X_data, h_X_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_classes, h_classes, dataset->num_instances() * dataset->num_instances() * sizeof(float), cudaMemcpyHostToDevice);

	//rectangular grid of 16x16 blocks

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	cudaEventRecord(start);

	int threadsPerBlockDim = 16;
	int gridDimSize = (dataset->num_instances() + threadsPerBlockDim - 1) / threadsPerBlockDim;

	dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
	dim3 gridSize (gridDimSize, gridDimSize);

	//stores into d_pairwise_distances the distance between instance i and instance j
	pairwiseDistanceKernel<<<gridSize, blockSize>>>(d_X_data, d_pairwise_distances, dataset->num_instances(), num_features);

	int threadsPerBlockDimReduction = 16;
	int gridDimSizeReduction = (dataset->num_instances() + threadsPerBlockDimReduction - 1) / threadsPerBlockDimReduction;

	dim3 blockSizeReduction(threadsPerBlockDimReduction);
	dim3 gridSizeReduction (gridDimSizeReduction);

	perInstanceKNN_LinearReduction<<<gridSizeReduction, blockSizeReduction>>>(d_pairwise_distances, d_classes, k,  dataset->num_instances());

	cudaMemcpy(h_classes, d_classes,
			dataset->num_instances() * dataset->num_instances() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU computation took %f ms\n", milliseconds);

	//compute max

	int* class_counts = (int*)calloc(NUM_CLASSES, sizeof(int));
	for(int instance=0; instance <  dataset->num_instances(); instance++){
		for(int i = 0; i < k; i++){
//			cout << h_classes[instance * dataset->num_instances() + i] << " ";
			class_counts[ (int) h_classes[instance * dataset->num_instances() + i] ] += 1;
		}

		int max = -1;
		int max_class = -1;
		for(int i = 0; i < NUM_CLASSES; i++){
			if(class_counts[i] > max){
				max = class_counts[i];
				max_class = i;
			}
		}
		predictions[instance] = max_class;
//		cout << " | " << predictions[instance];

//		cout << "\n";

		memset(class_counts, 0, NUM_CLASSES * sizeof(int));
	}

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }







//	for(int instance = 0; instance < 200; instance++){
//		for(int instance2 = 0; instance2 < dataset->num_instances(); instance2++){
//			cout << h_pairwise_distances[instance*dataset->num_instances() + instance2] << " ";
//		}
//		cout << "\n";
//	}


    cudaFree(d_X_data);
    cudaFree(d_classes);

	return predictions;
}



int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
//        cout << trueClass << " " << predictedClass << "\n";

        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[]){


    if(argc != 3)
    {
        cout << "Usage: ./main datasets/datasetFile.arff k" << endl;
        exit(0);
    }



    int k = strtol(argv[2], NULL, 10);

    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    int* predictions = NULL;

    // Get the class predictions
	predictions = KNN_GPU(dataset, k);

	// Compute the confusion matrix
	int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
	// Calculate the accuracy
	float accuracy = computeAccuracy(confusionMatrix, dataset);

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;


	printf("The %i-NN classifier for %lu instances required %llu ms CPU time on %i processes, accuracy was %.4f\n",
		   k,dataset->num_instances(), (long long unsigned int) diff, 1, accuracy);

}
