#ifndef _DEFINE_H_
#define _DEFINE_H_

#define PATH					"./Data/"

// #define DR_MODE					1	//test DR with one type of DR 
// #define SDR_MODE				1	//test SDR with one type of DR
//#define TAPKEE_NUM_DR			15	//no. of DR defined in tapkee (LandmarkMultidimensionalScaling 11, RandomProjection 15, tDistributedStochasticNeighborEmbedding 17)
#define TAPKEE_NUM_NEIGHBORS	10
//#define LANDMARK_RATIO			0.1

//#define LEARNING_RATE			0.1 //Main parameter of SDR
//#define NUM_ITR					10	//Parameter of SDR
#define KNN_K					50  //Parameter of SDR
#define EPSIL					0.0001
#define TSNE_PERPLEXITY			50 // default, set to 50 (tapkee default is 30)

typedef unsigned long long uint64;
typedef unsigned long uint32;
typedef unsigned short uint16;
typedef unsigned char uint8; 
#endif

/*
				tapkee::DimensionReductionMethod::KernelPCA 13							// Exclude: similar to LMDS
				tapkee::DimensionReductionMethod::LandmarkMultidimensionalScaling 11
				tapkee::DimensionReductionMethod::MultidimensionalScaling 10
				tapkee::DimensionReductionMethod::PCA 14								
				tapkee::DimensionReductionMethod::RandomProjection 15					
				tapkee::DimensionReductionMethod::tDistributedStochasticNeighborEmbedding 17
*/