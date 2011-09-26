//
// C++ Implementation: cforest
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include "cforest.h"

#include "cdatafactory.h"
#include "cextratrees.h"

CRegressionForest::CRegressionForest(int numTrees, int numDim) : CForest<double>(numTrees), CMapping<double>(numDim)
{
	
}
 
CRegressionForest::~CRegressionForest()
{
}
		
double CRegressionForest::doGetOutputValue(ColumnVector *vector)
{
	double average = 0;
	int numVal = 0;
	for (int i = 0; i < numTrees; i++)
	{
		if (forest[i] != NULL)
		{
			average += forest[i]->getOutputValue(vector);
			numVal ++;
		}
	}
	return average / numVal;
}

void CRegressionForest::saveASCII(FILE *stream)
{
	fprintf(stream, "%f %f\n", getAverageDepth(), getAverageNumLeaves());
	
}


CExtraTreeRegressionForest::CExtraTreeRegressionForest(int numTrees, CDataSet *inputData, CDataSet1D *outputData, unsigned int K,unsigned  int n_min, double treshold,  CDataSet1D *weightData) : CRegressionForest(numTrees, inputData->getNumDimensions())
{
	for (int i = 0; i < numTrees; i++)
	{
		addTree( i, new CExtraRegressionTree(inputData, outputData, K, n_min, treshold, weightData));		

	}
}

CExtraTreeRegressionForest::~CExtraTreeRegressionForest()
{
	for (int i = 0; i < numTrees; i++)
	{
		delete forest[i];
	}
}

double CRegressionMultiMapping::doGetOutputValue(ColumnVector *inputVector)
{
	double value = 0;

	for (int i = 0; i < numMappings; i ++)
	{
		value += mappings[i]->getOutputValue(inputVector);
	}
	value = value / numMappings;

	return value;
}

CRegressionMultiMapping::CRegressionMultiMapping(int l_numMappings, int numDimensions) : CMapping<double>(numDimensions)
{
	numMappings = l_numMappings;

	mappings = new CMapping<double> *[numMappings];

	deleteMappings = true;
}

CRegressionMultiMapping::~CRegressionMultiMapping()
{
	if (deleteMappings)
	{
		for (int i = 0; i < numMappings; i ++)
		{
			delete mappings[i];
		}
	}
	delete mappings;
}

void CRegressionMultiMapping::addMapping(int index, CMapping<double> *mapping)
{
	mappings[index] = mapping;
}
