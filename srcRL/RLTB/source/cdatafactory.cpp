//
// C++ Implementation: cdatafactory
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "cdatafactory.h"
#include "cinputdata.h"

CRegressionFactory::CRegressionFactory(CDataSet1D *l_outputData, CDataSet1D *weighting)
{
	outputData = l_outputData;
	weightData = weighting;
}
	
CRegressionFactory::~CRegressionFactory()
{
}
		
double CRegressionFactory::createTreeData(DataSubset *dataSubset, int ) 
{
	DataSubset::iterator it = dataSubset->begin();
	
	double mean = outputData->getMean( dataSubset, weightData);
		
	return mean;
}

DataSubset *CSubsetFactory::createTreeData(DataSubset *dataSubset, int ) 
{
	DataSubset *newSubSet = new DataSubset();
	
	DataSubset::iterator it = dataSubset->begin();
	
	for (; it != dataSubset->end(); it ++)
	{
		newSubSet->insert(*it);
	}
	
	return newSubSet;
}
	
void CSubsetFactory::deleteData(DataSubset *dataSet)
{
	delete dataSet;
}

CVectorQuantizationFactory::CVectorQuantizationFactory(CDataSet *l_inputData)
{
	inputData = l_inputData;
}
 
CVectorQuantizationFactory::~CVectorQuantizationFactory()
{
}
				
		
ColumnVector *CVectorQuantizationFactory::createTreeData(DataSubset *dataSubset, int ) 
{
	ColumnVector *mean = new ColumnVector(inputData->getNumDimensions());
	DataSubset::iterator it = dataSubset->begin();
	
	for (; it != dataSubset->end(); it ++)
	{
		(*mean) = (*mean) + *(*inputData)[*it];
	}
	(*mean) = (*mean) / dataSubset->size();
	
	return mean;	
}
 
void CVectorQuantizationFactory::deleteData(ColumnVector *dataSet)
{
	delete dataSet;
}


int CLeafIndexFactory::createTreeData(DataSubset *, int numLeaves)
{
	return numLeaves;
}
