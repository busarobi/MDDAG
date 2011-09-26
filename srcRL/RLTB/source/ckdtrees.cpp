#include "ckdtrees.h"

#include <algorithm>
#include <stdio.h>
#include <iostream>

#include "newmat/newmatio.h"

#include "cdatafactory.h"
#include "cinputdata.h"


CKDTreeMedianSplittingFactory::CKDTreeMedianSplittingFactory(CDataSet *l_inputSet, int l_n_min)
{
	n_min = l_n_min;
	inputData = l_inputSet;
}
	
CKDTreeMedianSplittingFactory::~CKDTreeMedianSplittingFactory()
{

}

CSplittingCondition *CKDTreeMedianSplittingFactory::createSplittingCondition(DataSubset *dataSubset)
{
	DataSubset::iterator it = dataSubset->begin(); 
	
	ColumnVector minVector(inputData->getNumDimensions());
	ColumnVector maxVector(inputData->getNumDimensions());
	
	ColumnVector *vector = (*inputData)[*it];
	minVector = *vector;
	maxVector = *vector;

	it ++;

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
	
	maxVector = maxVector - minVector;
	
	//cout << "Dimension Width: " << maxVector.t();


	int index = 0;
	maxVector.maximum1(index);
	index --;
	
	std::vector<double> median;
	
	it = dataSubset->begin(); 
	//printf("Values : ");
	double mean = 0.0;
	for (; it != dataSubset->end(); it ++)
	{	
		ColumnVector *vector = (*inputData)[*it];
//		median.push_back(vector->element(index));
//		printf("%f ", vector->element(index));
		mean += vector->element(index);
	}
	mean /= dataSubset->size();
	//std::sort(median.begin(), median.end());

	double treshold = mean; //median[median.size() / 2];

//	cout << "Dimension " << index << " Median " << treshold << endl;
	
	return new C1DSplittingCondition(index, treshold);
}

bool CKDTreeMedianSplittingFactory::isLeaf(DataSubset *dataSubset)
{
	bool minSampels = dataSubset->size() < (unsigned int) n_min;

	double inputVar = inputData->getVarianceNorm(dataSubset);
		
//	printf("IsLeaf: %d %f\n", dataSubset->size(), inputVar);
	return minSampels || inputVar <= 0.000001;
}

CKDTree::CKDTree(CDataSet *dataSet, int n_min) : CTree<DataSubset *>(dataSet->getNumDimensions())
{
	splittingFactory = new CKDTreeMedianSplittingFactory(dataSet, n_min);
	CTree<DataSubset *>::createTree(dataSet, splittingFactory, new CSubsetFactory());
	
}

CKDTree::~CKDTree()
{
	delete root;
	root = NULL;
	delete dataFactory;
	delete splittingFactory;
}

void CKDTree::addNewInput(int index)
{
	CTree<DataSubset *>::addNewInput(index, splittingFactory);
}