/*
 * maxoutFprop.c
 *
 *  Created on: Aug 24, 2014
 *      Author: orhanf
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mex.h"
#include "matrix.h"

#define TYPE double

/* Globals */
int     poolSize  =0;
int     stride	  =0;
int     isRandom  =0;
int     isDebug	  =0;
int 	nSamples  =0;
int		nFeatures =0;

/**
 * Usage
 */
void usage(void){
	mexErrMsgTxt("Usage: [pooledData, switches] = mexFunction(data, poolSize, stride, isRandom, isDebug);\n"
			"	data     : training mini-batch mapped to the max pooling layer (int)\n"
			"	poolSize : max pooling range, should divide number of neurons (int)\n"
			"	stride   : distance between consecutive max pooling ranges (int)\n"
			"	isRandom : flag indicating max pooling over randomized subsets of activations(0,1)\n"
			"	isDebug  : flag for verbose (0,1)\n");
}

/**
 * Prints a matrix that is row major order
 *
 * @param inputData
 * @param nFeatures
 * @param nSamples
 */
void print_matrix(TYPE* inputData, int nFeatures, int nSamples){
	int i,j;
	for(i=0 ; i<nFeatures ; ++i, mexPrintf("\n"))
		for(j=0 ; j<nSamples ; ++j)
			mexPrintf("%.4f ",inputData[ j*nFeatures+i]);
}

/**
 * Max pools the column of inputData matrix (vectorized)
 *
 * @param inputData
 * @param pooledData
 * @param switches
 */
void maxPool_1D(TYPE* inputData, TYPE* pooledData, int* switches){

	int nPools = nFeatures / poolSize;
	int i,j,k,startInd,ind = 0;
	double maxVal = -1e-32;

	for(i=0 ; i < nPools ; ++i)
		for(j=0 ; j < nSamples ; ++j){

			startInd = j*nFeatures+(i*poolSize);

			for(maxVal=-INFINITY, ind=0, k = startInd ; k < startInd+poolSize ;++k)
				if (inputData[k]>maxVal){
					maxVal = inputData[k];
					ind = k;
				}

			switches[j*nPools+i] = ind + 1;
			pooledData[j*nPools+i] = maxVal;
		}

}

/**
 * TODO not implemented yet
 *
 * @param inputData
 * @param pooledData
 * @param switches
 */
void maxPool_1D_random(TYPE* inputData, TYPE* pooledData, int* switches){}

/**
 * Interface to use Maxout-forward prop code in Matlab.
 *
 * @param nlhs  Number of expected mxArrays (Left Hand Side)
 * @param plhs 	Array of pointers to expected outputs
 * @param nrhs 	Number of inputs (Right Hand Side)
 * @param prhs 	Array of pointers to input data. The input data is read-only and should not be altered by your mexFunction .
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

	/* Variables */
	TYPE* 	inputData	=0;
	TYPE*   pooledData	=0;
	int*	switches	=0;
	int 	nFields		=0;

	/* Get input arguments 	*/
	poolSize = (int)mxGetScalar(prhs[1]);
	stride   = (int)mxGetScalar(prhs[2]);
	isRandom = (int)mxGetScalar(prhs[3]);
	isDebug  = (int)mxGetScalar(prhs[4]);

	/* Get starting address of real data in input array. */
	if (mxIsDouble(prhs[0]))
		inputData = (double *)mxGetPr(prhs[0]);
	else if (mxIsSingle(prhs[0]))
		inputData = (float *)mxGetPr(prhs[0]);
	else{
		usage();
		mexErrMsgTxt("First argument must be a either a double or float matrix.");
	}

	/* Get input dimensions */
	nFeatures = mxGetM(prhs[0]);
	nSamples  = mxGetN(prhs[0]);

	if(isDebug){
		mexPrintf("There are %d right-hand-side argument(s).\n", nrhs);
		mexPrintf("nFeatures :%d\n", nFeatures);
		mexPrintf("nSamples  :%d\n", nSamples);
		mexPrintf("poolSize  :%d\n", poolSize);
		mexPrintf("stride    :%d\n", stride);
		mexPrintf("isRandom  :%d\n", isRandom);
		mexPrintf("isDebug   :%d\n", isDebug);

		/* Display the contents of input matrix. */
		print_matrix(inputData, nFeatures, nSamples);
	}

	/* Allocation of output array 	*/
	pooledData = (TYPE *) mxGetPr(plhs[0] = mxCreateDoubleMatrix(nFeatures/poolSize, nSamples, mxREAL));
	switches   = (int *) mxGetPr(plhs[1] = mxCreateNumericMatrix(nFeatures/poolSize, nSamples, mxINT32_CLASS,mxREAL));

	/* Max-Pool input data, actual work here!	*/
	if (isRandom)
		maxPool_1D_random(inputData, pooledData, switches);
	else
		maxPool_1D(inputData, pooledData, switches);
}

