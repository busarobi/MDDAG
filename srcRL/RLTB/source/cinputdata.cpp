//
// C++ Implementation: cinputdata
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <assert.h>
#include "cinputdata.h"
#include <math.h>

void CDataPreprocessor::preprocessDataSet(CDataSet *dataSet)
{
	for (unsigned int i = 0; i < dataSet->size(); i ++)
	{
		preprocessInput((*dataSet)[i], (*dataSet)[i]);
	}
}

CMeanStdPreprocessor::CMeanStdPreprocessor(ColumnVector *l_mean, ColumnVector *l_std)
{
	mean = new ColumnVector(*l_mean);
	std = new ColumnVector(*l_std);
}

CMeanStdPreprocessor::CMeanStdPreprocessor(CDataSet *dataSet)
{
	mean = new ColumnVector(dataSet->getNumDimensions());
	std = new ColumnVector(dataSet->getNumDimensions());

	dataSet->getMean(NULL, mean);
	dataSet->getVariance( NULL, std);

	for (int i = 0; i < std->nrows(); i ++)
	{
		std->element(i) = sqrt(std->element(i));
	}
}

CMeanStdPreprocessor::~CMeanStdPreprocessor()
{
	delete mean;
	delete std;
}

void CMeanStdPreprocessor::preprocessInput(ColumnVector *input, ColumnVector *preInput)
{
	for (int i = 0; i < std->nrows(); i ++)
	{
		preInput->element(i) = (input->element(i) - mean->element(i)) / std->element(i);
	}
	
}

void CMeanStdPreprocessor::setMean(ColumnVector *l_mean)
{
	*mean = *l_mean;
}

void CMeanStdPreprocessor::setStd(ColumnVector *l_std)
{
	*std = *l_std;
}




CDataSet::CDataSet(int l_numDimensions)
{
	numDimensions = l_numDimensions;

	buffVector1 = new ColumnVector(numDimensions);
	buffVector2 = new ColumnVector(numDimensions);
}

CDataSet::CDataSet(CDataSet &dataset) : std::vector<ColumnVector *>()
{
	numDimensions = dataset.getNumDimensions();

	buffVector1 = new ColumnVector(numDimensions);
	buffVector2 = new ColumnVector(numDimensions);

	CDataSet::iterator it = dataset.begin();
	for (; it != dataset.end(); it ++)
	{
		addInput( *it);
	}
}
	
CDataSet::~CDataSet()
{
	CDataSet::iterator it = begin();
	
	for (; it != end(); it ++)
	{
		delete *it;
	}

	delete buffVector1;
	delete buffVector2;
}
		
int CDataSet::getNumDimensions()
{
	return numDimensions;
}

void CDataSet::clear()
{
	CDataSet::iterator it = begin();

	for (; it != end(); it ++)
	{
		delete *it;
	}
	std::vector<ColumnVector *>::clear();
}
				
void CDataSet::addInput(ColumnVector *input)
{
	ColumnVector *vector = new ColumnVector(*input);
	
	assert(vector->nrows() >= numDimensions);
	
	push_back(vector);
}


void CDataSet::saveCSV(FILE *stream)
{
	ColumnVector *vector;
	CDataSet::iterator it = begin();
	for (; it != end();it ++)
	{
		vector = *it;
		
		for (int i = 0; i < numDimensions; i ++)
		{
			fprintf(stream, "%f ", vector->element(i));
		}
		fprintf(stream, "\n");
	}
}
 
void CDataSet::loadCSV(FILE *stream)
{
	ColumnVector vector(getNumDimensions());
	while (!feof(stream))
	{
		int results = 0;
		for (int i = 0; i < numDimensions; i ++)
		{
			double buf = 0;
			results += fscanf(stream, "%lf ", &buf);
			vector.element(i) = buf;
		}
		fscanf(stream, "\n");
		if (results == numDimensions)
		{
			addInput(&vector);
		}
		else
		{
			printf("Loading Input Data: Wrong number  of Dimensions!!\n");
			break;
		}
	}
}
		
void CDataSet::getSubSet(DataSubset *subSet, CDataSet *newSet)
{
	DataSubset::iterator it = subSet->begin();
	
	for (; it != subSet->begin(); it ++)
	{
		newSet->addInput((*this)[(*it)]);
	}
	
}

double CDataSet::getVarianceNorm(DataSubset *dataSubset)
{
	getVariance(dataSubset, buffVector1);
	return buffVector1->norm_Frobenius();
}

void CDataSet::getVariance(DataSubset *dataSubset, ColumnVector *variance)
{
	ColumnVector *mean = buffVector1;
	ColumnVector *squaredMean = buffVector2;

	
	*mean = 0;	
	*squaredMean = 0;
	
	if (dataSubset != NULL)
	{
		DataSubset::iterator it = dataSubset->begin();
	
		for (; it != dataSubset->end(); it++)
		{
			ColumnVector *data = (*this)[*it];
			*mean += *data;
			
			(*squaredMean) = (*squaredMean) + SP(*data, *data);
		}
		(*mean) = (*mean) / dataSubset->size();
		(*squaredMean) = (*squaredMean)/ dataSubset->size();
	}
	else
	{
		for (unsigned int i = 0; i < size(); i++)
		{
			ColumnVector *data = (*this)[i];
			(*mean) += *data;
			
			(*squaredMean) = *squaredMean + SP(*data, *data);
		}
		(*mean) = *mean / size();
		(*squaredMean) = *squaredMean / size();
	}


	*mean = SP(*mean, *mean);
	*variance = *squaredMean - *mean;	
}


void CDataSet::getMean(DataSubset *dataSubset, ColumnVector *mean)
{
	*mean = 0;
	
	if (dataSubset)
	{	
		DataSubset::iterator it = dataSubset->begin();
			
		for (; it != dataSubset->end(); it++)
		{
			ColumnVector *data = (*this)[*it];
			*mean += *data;
			
		
		}
		(*mean) = *mean / dataSubset->size();
	}
	else
	{
		for (unsigned int i = 0; i < size(); i++)
		{
			ColumnVector *data = (*this)[i];
			*mean += *data;
		}
		(*mean) = *mean / size();
	
	}
}

CDataSet1D::CDataSet1D(CDataSet1D &dataset) : std::vector<double>(dataset)
{
	
}

CDataSet1D::CDataSet1D() : std::vector<double>()
{
	
}

void CDataSet1D::loadCSV(FILE *stream)
{
	while (!feof(stream))
	{
		int results = 0;
		
		double buf = 0;
		results += fscanf(stream, "%lf ", &buf);
			
		if (results == 1)
		{
			push_back(buf);
		}
		else
		{
			printf("Loading Input Data: Wrong number  of Dimensions!!\n");
			break;
		}
	}
}

void CDataSet1D::saveCSV(FILE *stream)
{
	CDataSet1D::iterator it = begin();
	for (; it != end();it ++)
	{	
		fprintf(stream, "%f ", *it);
		fprintf(stream, "\n");
	}
}

double CDataSet1D::getVariance(DataSubset *dataSubset, CDataSet1D *weighting)
{
	
	double mean = 0;
	double squaredMean = 0;

	double weightSum = 0;
	if (dataSubset == NULL)
	{
		for (unsigned int i = 0; i < size(); i ++)
		{
			double weight = 1.0;
	
			if (weighting != NULL)
			{
				weight = (*weighting)[i];
			}
			
			mean += weight * (*this)[i];
			squaredMean += weight * pow((*this)[i], 2.0);
	
			weightSum += weight;
		}
	}
	else
	{
		DataSubset::iterator it = dataSubset->begin();
	
		for (; it != dataSubset->end(); it++)
		{
			double weight = 1.0;
	
			if (weighting != NULL)
			{
				weight = (*weighting)[*it];
			}
			
			mean += weight * (*this)[*it];
			squaredMean += weight * pow((*this)[*it], 2.0);
	
			weightSum += weight;
		}
	}
	if (weightSum > 0)
	{
		mean /= weightSum;
		squaredMean /= weightSum;
	}
	
	return squaredMean - pow(mean, 2.0);
}

double CDataSet1D::getMean(DataSubset *dataSubset, CDataSet1D *weighting)
{
	
	double mean = 0;
	
	double weightSum = 0;

	if (dataSubset != NULL)
	{
		DataSubset::iterator it = dataSubset->begin();
	
		for (; it != dataSubset->end(); it++)
		{
			double weight = 1.0;
	
			if (weighting != NULL)
			{
				weight = (*weighting)[*it];
			}
			mean += weight * (*this)[*it];
			weightSum += weight;
		}
	}
	else
	{
		for (unsigned int i = 0; i < size(); i++)
		{
			double weight = 1.0;
	
			if (weighting != NULL)
			{
				weight = (*weighting)[i];
			}
			mean += weight * (*this)[i];
			weightSum += weight;
		}
	}

	if (weightSum > 0)
	{
		mean /= weightSum;
	}
	
	return mean;
}


void DataSubset::addElements(std::list<int> *subsetList)
{
	std::list<int>::iterator it = subsetList->begin();

	for (; it != subsetList->end(); it ++)
	{
		insert(*it);
	}
}
