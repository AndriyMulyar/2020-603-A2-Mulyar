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

__global__ void pairwiseDistanceKernel(float* X_data, float* distances){

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
	float* h_pairwise_distances  = (float*)malloc( num_elements * sizeof(float) );
	float* h_y_data = (float*)malloc(dataset->num_instances() * sizeof(float));

	for(int instance = 0; instance < dataset->num_instances(); instance++){
		h_y_data[instance] = dataset->get_instance(instance)->get(num_features)->operator float();
		for(int feature = 0; feature < num_features; feature++){
			h_X_data[instance*num_features + feature] = dataset->get_instance(instance)->get(feature)->operator float();
		}
	}

	float *d_X_data, *d_pairwise_distances, *d_y_data;

	cudaMalloc(&d_X_data, num_elements * sizeof(float));
	cudaMalloc(&d_pairwise_distances, num_elements * sizeof(float));
	cudaMalloc(&d_y_data, dataset->num_instances() * sizeof(float));

	cudaMemcpy(d_X_data, h_X_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_data, h_y_data, dataset->num_instances() * sizeof(float), cudaMemcpyHostToDevice);

	//rectangular grid of 16x16 blocks

	int threadsPerBlockDim = 16;
	int gridDimSizeX = (num_features + threadsPerBlockDim - 1) / threadsPerBlockDim; //will always be 1 for our datasets.
	int gridDimSizeY = (dataset->num_instances() + threadsPerBlockDim - 1) / threadsPerBlockDim;

	dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
	dim3 gridSize (gridDimSizeX, gridDimSizeY);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	pairwiseDistanceKernel<<<gridSize, blockSize>>>(d_X_data, d_pairwise_distances);

	cudaMemcpy(h_pairwise_distances, d_pairwise_distances, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);




	for(int instance = 0; instance < dataset->num_instances(); instance++){
		for(int feature = 0; feature < num_features; feature++){
			cout << h_pairwise_distances[instance*num_features + feature] << " ";
		}
//		cout << "|" << y_data[instance] << "\n";
	}


    cudaFree(d_X_data);
    cudaFree(d_y_data);

	return predictions;
}


//int compare(const void *a, const void *b) {
//    int x1 = *(const int*)a;
//    int x2 = *(const int*)b;
//    if (x1 > x2) return  1;
//    if (x1 < x2) return -1;
//    // x1 and x2 are equal; compare y's
//    int y1 = *(((const int*)a)+1);
//    int y2 = *(((const int*)b)+1);
//    if (y1 > y2) return  1;
//    if (y1 < y2) return -1;
//    return 0;
//}


//int* KNN_MPI(ArffData* dataset, int k){
//    //For each query vector, sends out instance id's to each worker thread to compute local k-NN on subset.
//    //Gathers the computed distances and classes from each worker process then re-ranks to generate global k-NN.
//    int rank, num_tasks;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
//
//    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
//
//    //Compute number of classes
//    int NUM_CLASSES = dataset->num_classes();
//    int NUM_INSTANCES = dataset->num_instances();
////    for(int i = 0; i < dataset->num_instances(); i++){
////        int class_index = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator long();
////        if(class_index+1 > NUM_CLASSES){
////            NUM_CLASSES = class_index+1;
////        }
////    }
//
//    //stores global k-NN candidates for a query vector, filled via gather from all processes
//    float* global_candidates = (float*) calloc(k*2*num_tasks, sizeof(float));
//
//    //stores per process local k-NN candidates as a sorted 2d array. First element is inner product, second is class.
//    float* local_candidates = (float*) calloc(k*2, sizeof(float));
//    for(int i = 0; i < 2*k; i++){local_candidates[i] = FLT_MAX;}
//
//    //stores bincounts of each class over the final set of candidate NN
//    int* classCounts = (int*)calloc(NUM_CLASSES, sizeof(int));
//
//    int instances_per_task = NUM_INSTANCES / num_tasks + 1;
//
//    int* displacements = (int *)malloc(num_tasks*sizeof(int));
//    int* receive_counts = (int *)malloc(num_tasks*sizeof(int));
//
//    for(int i = 0; i < num_tasks; i++){
//        displacements[i] = k*2*i;
//        receive_counts[i] = 2*k;
//    }
//
//    //Array of instance ids padded with -1 to account for last task
//    int* instance_ids_to_scatter = (int*)malloc((instances_per_task * num_tasks)*sizeof(int));
//    memset(instance_ids_to_scatter, -1, (instances_per_task * num_tasks) * sizeof(int));
//    for(int i = 0; i < NUM_INSTANCES; i++){
//        instance_ids_to_scatter[i] = i;
//    }
//
//    //Start MPI section
//
//    //contains per task instances
//    int* task_instances = (int*)malloc(instances_per_task*sizeof(int));
//
//    //the current query index (scattered by task 0 for every query)
//
//
//    //send instances to each process for computation
//    MPI_Scatter(instance_ids_to_scatter, instances_per_task, MPI_INT, task_instances, instances_per_task, MPI_INT,0, MPI_COMM_WORLD);
//
//    for(int query = 0; query < NUM_INSTANCES; query++){
//        for(int key = 0; key < instances_per_task; key++){
//            if (query == task_instances[key] or task_instances[key] == -1) continue;
//            float d_squared = distanceSquared(dataset->get_instance(query), dataset->get_instance(task_instances[key]));
////            cout << d_squared << endl;
//            for(int c = 0; c < k; c++){
//                if(d_squared < local_candidates[2*c]) {
//                    //Found a new candidate
//                    //Shift previous candidates down by one
//                    for (int x = k - 2; x >= c; x--) {
//                        local_candidates[2 * x + 2] = local_candidates[2 * x];
//                        local_candidates[2 * x + 3] = local_candidates[2 * x + 1];
//                    }
//                    //set key vector as potential k NN
//                    local_candidates[2 * c] = d_squared;
//                    local_candidates[2 * c + 1] = dataset->get_instance(task_instances[key])->get(
//                            dataset->num_attributes() - 1)->operator float();
//
//                    break;
//                }
//            }
//
//        }
//
//        //local_candidates now contains the local KNN for the query
//        //Gather back to the main process
//        MPI_Gatherv(local_candidates, 2*k, MPI_FLOAT,
//                    global_candidates, receive_counts, displacements, MPI_FLOAT, 0,
//                   MPI_COMM_WORLD);
//
//
//        //compute true k-NN from set of global candidates
//
//
//
//
//        if(rank == 0){
//            //sorted global candidates by distance
//            qsort(global_candidates, k*num_tasks, 2*sizeof(float), compare);
//
//            //bincount the candidate labels and pick the most common
//            for(int i = 0; i < k;i++){
//                classCounts[(int)global_candidates[2*i+1]] += 1;
//            }
//            int max = -1;
//            int max_index = 0;
//            for(int i = 0; i < NUM_CLASSES;i++){
//                if(classCounts[i] > max){
//                    max = classCounts[i];
//                    max_index = i;
//                }
//            }
//
//            predictions[query] = max_index;
//            memset(classCounts, 0, NUM_CLASSES * sizeof(int));
//
//        }
//
//
//        for(int i = 0; i < 2*k; i++){local_candidates[i] = FLT_MAX;}
//
//    }
//
//    return predictions;
//}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
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
