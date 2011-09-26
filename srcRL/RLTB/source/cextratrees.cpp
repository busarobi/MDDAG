//
// C++ Implementation: cextratrees
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "cextratrees.h"

#include "cdatafactory.h"
#include "cinputdata.h"

#include <math.h>

#include <iostream.h>
#include "newmat/newmatio.h"

CExtraTreesSplittingConditionFactory::CExtraTreesSplittingConditionFactory(CDataSet *l_inputData, CDataSet1D *l_outputData, unsigned int l_K, unsigned int l_n_min, double l_outTresh, CDataSet1D *l_weightingData)
{
	inputData = l_inputData;
	outputData = l_outputData;
	weightingData = l_weightingData;
	
	K = l_K;
	n_min = l_n_min;
	outTreshold = l_outTresh;
}

CExtraTreesSplittingConditionFactory::~CExtraTreesSplittingConditionFactory()
{
}

double CExtraTreesSplittingConditionFactory::getScore(CSplittingCondition *condition, DataSubset *dataSubset)
{
	DataSubset leftDataSubset;
	DataSubset rightDataSubset;
	
	DataSubset::iterator it = dataSubset->begin();
		
	for (; it != dataSubset->end(); it++)
	{
		ColumnVector *vector = (*inputData)[*it];
		if (condition->isLeftNode(vector))
		{
			leftDataSubset.insert(*it);
		}
		else
		{
			rightDataSubset.insert(*it);
		}
	}
	if (leftDataSubset.size() == 0 || rightDataSubset.size() == 0)
	{
		return 1.0;
	}

		

	return (- outputData->getVariance( &leftDataSubset, weightingData) * leftDataSubset.size() - outputData->getVariance( &rightDataSubset, weightingData) * rightDataSubset.size()) / dataSubset->size();

}
		
 
CSplittingCondition *CExtraTreesSplittingConditionFactory::createSplittingCondition(DataSubset *dataSubset)
{
	ColumnVector minVector(inputData->getNumDimensions());
	ColumnVector maxVector(inputData->getNumDimensions());
	
	DataSubset::iterator it = dataSubset->begin();
	
	
	ColumnVector *vector = (*inputData)[*it];
	minVector = *vector;
	maxVector = *vector;
	
	for (; it != dataSubset->end(); it ++)
	{
		ColumnVector *vector = (*inputData)[*it];
		
		for (int i = 0; i < inputData->getNumDimensions(); i ++)
		{
			if (vector->element(i) < minVector.element(i))
			{
				minVector.element(i) = vector->element(i);
			}
			if (vector->element(i) > maxVector.element(i))
			{
				maxVector.element(i) = vector->element(i);
			}			
		}
	}
	double bestScore = 0.0;
	CSplittingCondition *bestSplit = NULL;
	unsigned int i = 0;	

	int numValid = 0;
	while (i < K || numValid == 0)
	{
		int dim = rand() % inputData->getNumDimensions();
		
		double treshold = (((double) rand()) / RAND_MAX) * (maxVector.element(dim) - minVector.element(dim)) + minVector.element(dim);
		
		CSplittingCondition *newSplit = new C1DSplittingCondition(dim, treshold);
		
		double score = getScore( newSplit, dataSubset);

		if ((score > bestScore || numValid == 0) && score <= 0.5)
		{
			bestScore = score;
			
			if (bestSplit != NULL)
			{
				delete bestSplit;
			}
			bestSplit = newSplit;
		}
		else
		{
			delete newSplit;
		}
		if (score <= 0.5)
		{
			numValid ++;
		}
		i ++;
		if (i > 100)
		{
			//printf("%d: %d %f\n", i, dataSubset->size(), score);
		}
		if (i > 200)
		{
			it = dataSubset->begin();
			printf("Could not find a split for : \n");
			for (; it != dataSubset->end(); it++)
			{
				cout << (*inputData)[*it]->t();
			}
			
			exit(0);
		}
	}
	return bestSplit;
}
	
bool CExtraTreesSplittingConditionFactory::isLeaf(DataSubset *dataSubset)
{
	bool minSampels = dataSubset->size() < n_min;

	bool outputVar = outputData->getVariance(dataSubset, weightingData) < outTreshold;
	
	double inputVar = inputData->getVarianceNorm(dataSubset);
		
	
	bool leaf = minSampels || outputVar || inputVar <= 0.0001;

	/*if (leaf)
	{
		printf("Leaf: Input Variance: %f (%d, %d)\n", inputVar, dataSubset->size(), n_min);
	
	}
	else
	{
		printf("Input Variance: %f (%d)\n", inputVar, dataSubset->size());
	}*/

	return leaf;
}

CExtraRegressionTree::CExtraRegressionTree(CDataSet *inputData, CDataSet1D *outputData, unsigned int K, unsigned int n_min, double treshold, CDataSet1D *weightData) : CExtraTree<double>(inputData, outputData, new CRegressionFactory(outputData, weightData), K, n_min, treshold, weightData)
{
	
}

CExtraRegressionTree::~CExtraRegressionTree()
{
	delete root;
	root = NULL;
	delete dataFactory;
}

