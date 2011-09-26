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

#ifndef C_UTILITY_H
#define C_UTILITY_H

#include <map>
#include <assert.h>

#include <newmat/newmat.h>


//class Matrix;
//class ColumnVector;

int my_round(double value);

double my_exp(double value);

void getPseudoInverse(Matrix *J, Matrix *pinv, double lambda);

/*
class ColumnVector 
{
protected:
	double *data;
	unsigned int dimensions;

public:
	ColumnVector(unsigned int dimension, double *data = NULL);
	virtual ~ColumnVector();

	void addVector(ColumnVector *vector);
	void addScalar(double scalar);

	void dotVector(ColumnVector *vector);

	void multMatrix(Matrix *matrix, ColumnVector *output);
	void multMatrix(Matrix *matrix);

	void multVector(ColumnVector *vector, Matrix *output);
	
	void multScalar(double scalar);

	double getLength();
	void normalizeVector();
	
	double scalarProduct(ColumnVector *vector);
	virtual double getDistance(ColumnVector *vector);

	double element(unsigned int index);
	void setElement(unsigned int index, double value);

	unsigned int getNumDimensions();
//	bool isColumnVector();

	double *getData();

	void setVector(ColumnVector *vector);
	void initVector(double init);

	void saveASCII(FILE *stream);
	void loadASCII(FILE *stream);
};

class Matrix
{
protected:
	double *data;
	unsigned int columns;
	unsigned int rows;

public:
	Matrix(unsigned int rows, unsigned int columns, double *data = NULL);
	~Matrix();

	void addVector(ColumnVector *vector);
	void addScalar(double scalar);

	void multMatrix(Matrix *matrix, Matrix *output);

	void multVector(ColumnVector *vector, ColumnVector *output);

	void multScalar(double scalar);

	double element(unsigned int row, unsigned int column);
	void setElement(unsigned int row, unsigned int column, double value);

	double *getRow(unsigned int row);

	unsigned int nrows();
	unsigned int ncols();

	double *getData();

	void setMatrix(Matrix *matrix);
	void initMatrix(double init);

	void saveASCII(FILE *stream);
};*/

/// A multidimensional Array
/**
Stores an one dimensional array of Type T1, and provides the simulation of a multi-dimensional array.
The number of dimensions and the size of each dimension is given to the constructor. There are accessing functions
for one dimensional an multidimensional indices.

*/
template <typename T1> class CMyArray 
{
protected:
	T1 *data;

	int *dim;
	int numDim;

	int size;

	CMyArray() {};

	void initialize(int numDim, int dim[])
	{
		this->numDim = numDim;
		this->dim = new int[numDim];

		memcpy(this->dim, dim, numDim * sizeof(int));

		size = 1;

		for (int i = 0; i < numDim; i++)
		{
			size = size * dim[i];
		}

		data = new T1 [size];
	}

public:
	CMyArray(int numDim, int dim[])
	{
		CMyArray<T1>::initialize(numDim, dim);
	}

	~CMyArray()
	{
		delete dim;
		delete data;
	}

	T1 get(int indices[])
	{
		int index = 0, size = 1;

		for (int i = 0; i < numDim; i++)
		{
			assert(indices[i] < dim[i] && indices[i] >=0);

			index += indices[i] * size;
			size = size * dim[i];
		}
		return data[index];
	}

	void set(int indices[], T1 d)
	{
		int index = 0, size = 1;

		for (int i = 0; i < numDim; i++)
		{
			assert(indices[i] < dim[i] && indices[i] >=0);

			index += indices[i] * size;
			size = size * dim[i];
		}
		data[index] = d;
	
	}

	void init(T1 initVal)
	{
		for (int i = 0; i < size; i++)
		{
			data[i] = initVal;
		}
	}

	int getSize()
	{
		return size;
	}

	void set1D(int index1d, T1 d)
	{
		data[index1d] = d;
	}

	T1 get1D(int index1d)
	{
		return data[index1d];
	}
};

/// Collection of Distributions
class CDistributions
{
public:
///Returns the gibs distribution (soft max)
/** beta is the head, used by the formular  In the values array there are given the single values. The function
writes then the propability of the indices in the array.
*/
	static void getGibbsDistribution(double beta, double *values, unsigned int numValues);

	static void getS1L0Distribution(double *values, unsigned int numValues);

	static double getNormalDistributionSample(double mu, double sigma);

	static int getSampledIndex(double *distribution, int numValues);
};



/// 2 dimensional Array
template<typename T1> class CMyArray2D : public CMyArray<T1>
{
public:
	CMyArray2D(int xDim, int yDim) 
	{
		int *l_dim = new int[2];
		l_dim[0] = xDim;
		l_dim[1] = yDim;

		CMyArray<T1>::initialize(2, l_dim);

		delete l_dim;
	}

	~CMyArray2D()
	{
	}

	T1 get(int xIndex, int yIndex)
	{
		int indices[2];
		indices[0] = xIndex;
		indices[1] = yIndex;

		return CMyArray<T1>::get(indices);
	}

	void set(int xIndex, int yIndex, T1 d)
	{
		int indices[2];
		indices[0] = xIndex;
		indices[1] = yIndex;

		CMyArray<T1>::set(indices, d);
	}
};

/// 3 dimensional array
template<typename T1> class CMyArray3D : public CMyArray<T1>
{
public:
	CMyArray3D(int xDim, int yDim, int zDim) : CMyArray<T1>()
	{
		int *ldim = new int[3];
		ldim[0] = xDim;
		ldim[1] = yDim;
		ldim[2] = zDim;

		CMyArray<T1>::initialize(3, ldim);
	}

	~CMyArray3D()
	{
	}

	T1 get(int xIndex, int yIndex, int zIndex)
	{
		int indices[3];
		indices[0] = xIndex;
		indices[1] = yIndex;
		indices[2] = zIndex;

		return CMyArray<T1>::get(indices);
	}

	void set(int xIndex, int yIndex, int zIndex, T1 d)
	{
		int indices[3];
		indices[0] = xIndex;
		indices[1] = yIndex;
		indices[2] = zIndex;

		CMyArray<T1>::set(indices, d);
	}
};

// Maps feature indices to feature factors
class CFeatureMap : public std::map<int, double>
{
protected:
	double stdValue;

public:
	CFeatureMap(double stdValue = 0.0);

	double getValue(unsigned int featureIndex);
};
/*
/// N-dimensional Sparse for features
class CFeatureSparse
{
protected:
	CMyArray<CFeatureList *> *sparse;
	double stdValue;
	char *sparseName;

	int numDim;
	int *dim;

	void initSparse(int numDim, int dim[]);

	CFeatureSparse();

public:
	CFeatureSparse(FILE *file, int numDim = 0, int *dim = NULL);
	CFeatureSparse(int numDim, int dim[]);
	virtual ~CFeatureSparse();

	virtual double getFactor(int indeces[], unsigned int featureIndex);
	void setFactor(double factor, int indeces[], unsigned int featureIndex);
	void addFactor(double factor, int indeces[], unsigned int featureIndex);
	void loadASCII(FILE *stream);

	CFeature *getCFeature(int indeces[], unsigned int featureIndex);
	CFeature *getCFeature1D(int index1D, unsigned int featureIndex);
	virtual CFeatureList *getFeatureList(int indeces[]);
	
	void saveASCII(FILE *stream);
};

/// 2 dimensional feature sparse.
//so there a a 2 dimensional array of feature lists.

class CFeatureSparse2D : public CFeatureSparse
{
public:
	CFeatureSparse2D(FILE *file);
	CFeatureSparse2D(int dim1, int dim2);
	virtual ~CFeatureSparse2D();

	virtual double getFactor(int ind1, int ind2, unsigned int featureIndex);
	virtual void setFactor(double factor, int ind1, int ind2, unsigned int featureIndex);
	virtual void addFactor(double factor, int ind1, int ind2, unsigned int featureIndex);

	virtual CFeatureList *getFeatureList(int ind1, int ind2);

	CFeature* getCFeature(int ind1, int ind2, unsigned int featureIndex);

	
};*/

#endif
