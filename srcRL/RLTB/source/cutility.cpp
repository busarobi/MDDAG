// Copyright (C) 2003
// Gerhard Neumann (gneumann@gmx.net)
// Stephan Neumann (sneumann@gmx.net) 
//                
// This file is part of RL Toolbox.
// http://www.igi.tugraz.at/ril_toolbox
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ril_debug.h"
#include "cutility.h"
#include <assert.h>
#include <math.h>

#include <newmat/newmat.h>
#include <newmat/newmatap.h>

int my_round(double value)
{
	int intvalue = (int) floor(value);
	if (value - intvalue > 0.5)
	{
		intvalue ++;
	}
	return intvalue;
}

double my_exp(double value)
{
	if (value > MAX_EXP)
	{
		value = MAX_EXP;
	}
	else
	{
		if (value < MIN_EXP)
		{
			value = MIN_EXP;
		}
	}
	return exp(value);
}

void getPseudoInverse(Matrix *J, Matrix *pinv, double lambda )
{
	
	
	bool transpose = false;
	int n,m;
	n = m = 0;
	if (J->nrows() < J->ncols())
	{
		n = J->nrows();
		m = J->ncols();
		
		transpose = true;
	}
	else
	{
		m = J->nrows();
		n = J->ncols();
	}
	
	//Matrix A(m,n);
	
	Matrix U(m, n);
	Matrix V(n, n);
	
	DiagonalMatrix D(n);
	
	//printf("Calculating SVD... %d x %d\n", J->nrows(), J->ncols());
	if (transpose)
	{
		SVD(J->t(), D, U, V);	
	}
	else
	{
		SVD(*J, D, U, V);	
	}
	//printf("Done... \n");
	
	(*pinv) = 0;
	
	for (int i = 0; i < n; i++)
	{
		double s = D.element(i,i);
			
		if (transpose)
		{
			(*pinv) = (*pinv) + s / (s * s + lambda * lambda) * U.column(i + 1) * V.column(i + 1).t();	
		}
		else
		{
			(*pinv) = (*pinv) + s / (s * s + lambda * lambda) * V.column(i + 1) * U.column(i + 1).t();
		}
		
	}
}

/*
ColumnVector::ColumnVector(unsigned int dimensions, double *data)
{
	this->dimensions = dimensions;
	this->data = new double[dimensions];
	
	if (data)
	{
		memcpy(this->data, data, dimensions * sizeof(double));
	}
	else
	{
		for (unsigned int i = 0; i < dimensions; i++)
		{
			this->data[i] = 0.0;
		}
	}
}

ColumnVector::~ColumnVector()
{
	delete [] data;
}

void ColumnVector::dotVector(ColumnVector *vector)
{
	double *data_vec =  vector->getData();
	for (unsigned int i = 0;i < nrows(); i++)
	{
		data[i] *= data_vec[i];
	}
}

double ColumnVector::scalarProduct(ColumnVector *vector)
{
	assert(nrows() == vector->nrows());
	double *data2 = vector->getData();
	double skalarprod = 0.0;

	for (unsigned int i = 0; i < nrows(); i++)
	{
		skalarprod += data[i] * data2[i];
	}
	return skalarprod;
}

double ColumnVector::getDistance(ColumnVector *vector)
{
	assert(nrows() == vector->nrows());
	double *data2 = vector->getData();
	double distance = 0.0;

	for (unsigned int i = 0; i < nrows(); i++)
	{
		distance += pow(data[i] - data2[i], 2);
	}
	return sqrt(distance);
}

void ColumnVector::addVector(ColumnVector *vector)
{
	assert(nrows() == vector->nrows());
	double *data2 = vector->getData();
	
	for (unsigned int i = 0; i < nrows(); i++)
	{
		data[i] += data2[i];
	}
}

void ColumnVector::addScalar(double factor)
{
	for (unsigned int i = 0; i < nrows(); i++)
	{
		data[i] += factor;
	}
}

void ColumnVector::multMatrix(Matrix *matrix, ColumnVector *output)
{
	assert(nrows() == matrix->nrows() && output->nrows() == matrix->ncols());
	double *dataRow;
	double *dataOutput = output->getData();
	
	double *vectorData;
	bool newVec = false;

	if (dataOutput == data)
	{
		vectorData = new double[nrows()];
		memcpy(vectorData, data, nrows() * sizeof(double));
		newVec = true;
	}
	else
	{
		vectorData = data;
	}

	output->initVector(0.0);

	for (unsigned int i = 0; i < matrix->nrows(); i++)
	{
		dataRow = matrix->getRow(i);

		for (unsigned int j = 0; j < matrix->ncols(); j++)
		{
			dataOutput[j] += dataRow[j] * vectorData[i];
		}
	}
	if (newVec)
	{
		delete vectorData;
	}
}

void ColumnVector::multMatrix(Matrix *matrix)
{
	assert(nrows() == matrix->ncols() && nrows() == matrix->nrows());

    multMatrix(matrix, this);
}


void ColumnVector::multVector(ColumnVector *vector, Matrix *output)
{
	assert(nrows() == output->nrows() && vector->nrows() == output->ncols());
	double *data2 = vector->getData();
	double *dataRow = NULL;

	for (unsigned int j = 0; j < output->nrows(); j++)
	{
		dataRow = output->getRow(j);
		for (unsigned int i = 0; i < output->ncols(); i++)
		{
			dataRow[i] = data[j] * data2[i];
		}
	}
}

void ColumnVector::multScalar(double scalar)
{
	for (unsigned int i = 0; i < nrows(); i++)
	{
		data[i] *= scalar;
	}
}

double ColumnVector::element(unsigned int index)
{
	assert(index < nrows());
	return data[index];
}

void ColumnVector::saveASCII(FILE *stream)
{
	fprintf(stream,"[ ");
	for(unsigned int i = 0; i < nrows(); i++)
	{
		fprintf(stream, "%f ", data[i]);
	}
	fprintf(stream, "]\n");
}

void ColumnVector::loadASCII(FILE *stream)
{
	fscanf(stream,"[ ");
	for(unsigned int i = 0; i < nrows(); i++)
	{
		fscanf(stream, "%lf ", &data[i]);
	}
	fscanf(stream, "]\n");
}

void ColumnVector::setElement(unsigned int index, double value)
{
	assert(index < nrows());
	data[index] = value;
}

unsigned int ColumnVector::nrows()
{
	return dimensions;
}

double *ColumnVector::getData()
{
	return data;
}

void ColumnVector::setVector(ColumnVector *vector)
{
	assert(vector->nrows() == nrows());
	memcpy(data, vector->getData(), nrows() * sizeof(double));
}

void ColumnVector::initVector(double init)
{
	for (unsigned int i = 0; i < nrows(); i++)
	{
		data[i] = init;
	}
}

double ColumnVector::getLength()
{
	double length = 0;
	for (unsigned int i = 0; i < nrows(); i++)
	{
		length += pow(data[i], 2);
	}
	return sqrt(length);
}

void ColumnVector::normalizeVector()
{
	double length = getLength();
	if (length > 0.00001)
	{
		multScalar(1.0 / length);
	}
}


Matrix::Matrix(unsigned int rows, unsigned int columns, double *data)
{
	this->rows = rows;
	this->columns = columns;
	this->data = new double[rows * columns];

	if (data != NULL)
	{
		memcpy(this->data, data, rows * columns * sizeof(double));
	}
}

Matrix::~Matrix()
{
	delete data;
}

void Matrix::addVector(ColumnVector *vector)
{
	double *rowData;
	
	for (unsigned int i = 0; i < nrows(); i++)
	{
		rowData = getRow(i);
		for (unsigned int j = 0; j < ncols(); j++)
		{
			rowData[j] += vector->element(i);
		}
	}
}

void Matrix::addScalar(double scalar)
{	
	double *rowData;

	for (unsigned int i = 0; i < nrows(); i++)
	{
		rowData = getRow(i);
		for (unsigned int j = 0; j < ncols(); j++)
		{
			rowData[j] += scalar;
		}
	}
}


void Matrix::multMatrix(Matrix *matrix, Matrix *output)
{
	assert(this->ncols() == matrix->nrows() && output->nrows() == this->nrows() && output->ncols() == matrix->ncols());

	double *rowData = NULL;

	for (unsigned int i = 0; i < nrows(); i++)
	{
		rowData = output->getRow(i);
		for (unsigned int j = 0; j < ncols(); j++)
		{
			rowData[j] = 0.0;
			for (unsigned int k = 0; k < ncols(); k++)
			{
				rowData[j] += element(i, k) * matrix->element(k, j);
			}
		}
	}
}

void Matrix::multVector(ColumnVector *vector, ColumnVector *output)
{
	assert(vector->nrows() == ncols() && output->nrows() == nrows());
	double *dataRow;
	double *dataOutput = output->getData();

	double *vectorData = vector->getData();

	for (unsigned int i = 0; i < nrows(); i++)
	{
		dataRow = getRow(i);
		dataOutput[i] = 0.0;
		for (unsigned int j = 0; j < ncols(); j++)
		{
			dataOutput[i] += dataRow[j] * vectorData[j];
		}
	}
}

void Matrix::multScalar(double scalar)
{
	double *rowData;

	for (unsigned int i = 0; i < nrows(); i++)
	{
		rowData = getRow(i);
		for (unsigned int j = 0; j < ncols(); j++)
		{
			rowData[j] *= scalar;
		}
	}
}

double Matrix::element(unsigned int row, unsigned int column)
{
	return data[row * ncols() + column];
}

void Matrix::setElement(unsigned int row, unsigned int column, double value)
{
	data[row * ncols() + column] = value;
}

double *Matrix::getRow(unsigned int row)
{
	return (data + row * ncols());
}

double *Matrix::getData()
{
	return data;
}

unsigned int Matrix::nrows()
{
	return rows;
}

unsigned int Matrix::ncols()
{
	return columns;
}

void Matrix::setMatrix(Matrix *matrix)
{
	assert(matrix->nrows() == nrows() && matrix->ncols() == ncols());
	memcpy(data, matrix->getData(), ncols() * nrows() * sizeof(double));
}

void Matrix::initMatrix(double init)
{
	for (unsigned int i = 0; i < nrows() * ncols(); i++)
	{
		data[i] = init;
	}
}

void Matrix::saveASCII(FILE *stream)
{
	fprintf(stream,"[ ");
	for(unsigned int i = 0; i < nrows(); i++)
	{
		for (unsigned int j = 0; j < ncols(); j ++)
		{
			fprintf(stream, "%f ", element(i, j));
		}
		fprintf(stream, "\n");
	}
	fprintf(stream, "]");
}

*/

void CDistributions::getGibbsDistribution(double beta, double *values, unsigned int numValues)
{
	double sum = 0.0;
	unsigned int i;
	for (i = 0; i < numValues; i++)
	{
		values[i] = my_exp(beta * values[i]);
		sum += values[i];
	}
	for (i = 0; i < numValues; i++)
	{
		values[i] = values[i] / sum;
	}
}


void CDistributions::getS1L0Distribution(double *values, unsigned int numValues)
{
	double smallest, largest, sum;
	unsigned int i;
    //transform: smallest Value = 0, Valuesum = 1;
	smallest = sum = largest = values[0];
	for(i = 1; i < numValues; i++)
	{
		if (smallest > values[i]) smallest = values[i];
		if (largest < values[i]) largest = values[i];
		sum += values[i];
	}
    sum -=  numValues * smallest;
	if (largest == smallest) // alle Werte gleich ==> Werte so setzen, das alle Values das Gewicht 1 / numValues bekommen)
	{
		sum = numValues;
		smallest = smallest - 1;
	}
    for(i = 0; i < numValues; i++)
	{
		values[i] = (values[i] - smallest) / sum;
	}
}

double CDistributions::getNormalDistributionSample(double mu, double sigma)
{
	double x1 = (double) (rand() + 1) / ((double) RAND_MAX + 1);
	double x2 = (double) rand() / (double) RAND_MAX;

	double z = sqrt(- 2 * log(x1)) * cos(2 * M_PI * x2);

	return z * sqrt(sigma) + mu;
}

int CDistributions::getSampledIndex(double *distribution, int numValues)
{
	double z = (double) (rand()) / RAND_MAX;
	double sum = distribution[0];

	int index = 0;

	while (sum <= z && index < numValues - 1)
	{
		index++; 
		sum += distribution[index];
	}
	return index;
}



/*
CFeatureSparse::CFeatureSparse(FILE *file, int numDim, int *dim) 
{
	this->stdValue = 0.0;
	
	sparse = NULL;

	this->numDim = numDim;
	this->dim = dim;

	loadASCII(file);	
}

CFeatureSparse::CFeatureSparse(int numDim, int dim[])
{
	this->stdValue = 0.0;

	sparse = NULL;

	numDim = 0;
	dim = NULL;

	initSparse(numDim, dim);
}

CFeatureSparse::CFeatureSparse()
{
	this->stdValue = 0.0;

	sparse = NULL;

	numDim = 0;
	dim = NULL;
}

CFeatureSparse::~CFeatureSparse()
{
	CFeatureList *featList = NULL;

	for (int i = 0; i < sparse->getSize(); i++)
	{
		featList = sparse->get1D(i);

		featList->clearAndDelete();
			
		delete featList;
	}
	delete sparse;

	delete dim;
}

void CFeatureSparse::initSparse(int numDim, int dim[])
{
	this->numDim = numDim;
	this->dim = new int[numDim];

	memcpy(this->dim, dim, sizeof(int) * numDim);

	sparse = new CMyArray<CFeatureList *>(numDim, this->dim);
	
	for (int i = 0; i < sparse->getSize(); i++)
	{
		sparse->set1D(i, new CFeatureList());
	}
}

void CFeatureSparse::setFactor(double factor, int indeces[], unsigned int featureIndex)
{
	CFeature *feat = getCFeature( indeces, featureIndex);

	assert(feat->featureIndex == featureIndex);

	feat->factor = factor;
}

void CFeatureSparse::addFactor(double factor, int indeces[], unsigned int featureIndex)
{
	CFeature *feat = getCFeature( indeces,  featureIndex);

	assert(feat->featureIndex == featureIndex);

	feat->factor += factor;
}

CFeatureList *CFeatureSparse::getFeatureList(int indeces[])
{
	return sparse->get(indeces);
}

CFeature* CFeatureSparse::getCFeature(int indeces[], unsigned int featureIndex)
{
	CFeature *feat = NULL;

	CFeatureList *featList = sparse->get(indeces);

	CFeatureList::iterator it = featList->begin();

	while (it != featList->end() && (*it)->featureIndex < featureIndex)
	{
		it ++;
	}

	if (it != featList->end() && (*it)->featureIndex == featureIndex)
	{
		feat = *it;
	}
	else
	{
		feat = new CFeature(featureIndex, stdValue);
		featList->insert(it, feat);
	}
	return feat;
}

CFeature* CFeatureSparse::getCFeature1D(int index1D, unsigned int featureIndex)
{
	CFeature *feat = NULL;

	CFeatureList *featList = sparse->get1D(index1D);

	CFeatureList::iterator it = featList->begin();

	while (it != featList->end() && (*it)->featureIndex < featureIndex)
	{
		it ++;
	}

	if (it != featList->end() && (*it)->featureIndex == featureIndex)
	{
		feat = *it;
	}
	else
	{
		feat = new CFeature(featureIndex, stdValue);
		featList->insert(it, feat);
	}
	return feat;
}


double CFeatureSparse::getFactor(int indeces[], unsigned int featureIndex)
{
	double factor = stdValue;

	CFeatureList *featList = sparse->get(indeces);
	CFeatureList::iterator it = featList->begin();

	while (it != featList->end() && (*it)->featureIndex < featureIndex)
	{
		it ++;
	}

	if (it != featList->end() && (*it)->featureIndex == featureIndex)
	{
		factor = (*it)->factor;
	}
	return factor;
}

void CFeatureSparse::loadASCII(FILE *stream)
{
	int bufIndex, bufFeatures, numDim, i;
	double bufFactor;
	int *dim;
	
	fscanf(stream,"%dD-Sparse: [", &numDim);

	dim = new int[numDim];

	for (i = 0; i < numDim; i++)
	{
		fscanf(stream, "%d ", &dim[i]);
	}

	fscanf(stream, "]\n");

	if (this->numDim > 0)
	{
		assert(this->numDim == numDim);
		if (this->dim != NULL)
		{
			for (int i = 0; i < numDim; i++)
			{
				assert(dim[i] == this->dim[i]);
			}
		}
	}
	else
	{
		initSparse(numDim, dim);
	}

	for (i = 0; i < sparse->getSize(); i++)
	{
		fscanf(stream, "%d: ", &bufFeatures);

		for (int k = 0; k < bufFeatures; k++)
		{
			fscanf(stream, "(%d %lf)", &bufIndex, &bufFactor);

			getCFeature1D(i, bufIndex)->factor = bufFactor;
		}
		fscanf(stream, "\n");
	}
}

void CFeatureSparse::saveASCII(FILE *stream)
{
	CFeatureList *featList;
	CFeatureList::iterator it;
	int i;

	fprintf(stream,"%dD-Sparse: [", numDim);
	
	for (i = 0; i < numDim; i++)
	{
		fprintf(stream, "%d ", dim[i]);
	}
    fprintf(stream, "]\n");

	for (i = 0; i < sparse->getSize(); i++)
	{
		featList = sparse->get1D(i);

		fprintf(stream, "%d: ", featList->size());

		for (it = featList->begin(); it != featList->end(); it++)
		{
			fprintf(stream, "(%d %f)", (*it)->featureIndex, (*it)->factor);
		}
		fprintf(stream, "\n");
	}
}

CFeatureSparse2D::CFeatureSparse2D(FILE *file) : CFeatureSparse(file, 2)
{
}

CFeatureSparse2D::CFeatureSparse2D(int dim1, int dim2) : CFeatureSparse()
{
	int dim[2];
	dim[0] = dim1;
	dim[1] = dim2;

	initSparse(2, dim);
}

CFeatureSparse2D::~CFeatureSparse2D()
{
}

void CFeatureSparse2D::setFactor(double factor, int ind1, int ind2, unsigned int featureIndex)
{
	int indeces[2];
	indeces[0] = ind1;
	indeces[1] = ind2;

	CFeatureSparse::setFactor(factor, indeces, featureIndex);
}

void CFeatureSparse2D::addFactor(double factor, int ind1, int ind2, unsigned int featureIndex)
{
	int indeces[2];
	indeces[0] = ind1;
	indeces[1] = ind2;

	CFeatureSparse::addFactor(factor, indeces, featureIndex);
}

CFeatureList *CFeatureSparse2D::getFeatureList(int ind1, int ind2)
{
	int indeces[2];
	indeces[0] = ind1;
	indeces[1] = ind2;

	return sparse->get(indeces);
}

CFeature* CFeatureSparse2D::getCFeature(int ind1, int ind2, unsigned int featureIndex)
{
	int indeces[2];
	indeces[0] = ind1;
	indeces[1] = ind2;

	return CFeatureSparse::getCFeature(indeces, featureIndex);
}

double CFeatureSparse2D::getFactor(int ind1, int ind2, unsigned int featureIndex)
{
	int indeces[2];
	indeces[0] = ind1;
	indeces[1] = ind2;

	return CFeatureSparse::getFactor(indeces, featureIndex);
}
*/
CFeatureMap::CFeatureMap(double stdValue )
{
	this->stdValue = stdValue;
}

double CFeatureMap::getValue(unsigned int featureIndex)
{
	CFeatureMap::iterator reward = find(featureIndex);

	if (reward != end())
	{
		return (*reward).second;
	}
	else
	{
		return stdValue;
	}
}
