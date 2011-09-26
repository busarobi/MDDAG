#ifndef C_LOCALREGRESSION__H
#define C_LOCALREGRESSION__H

#include "cinputdata.h"


#include "newmat/newmatio.h"

#include <iostream>
#include <limits> 

class CDataSet;
class CDataSet1D;
class CDataSubSet;

class CKDTree;
class CKNearestNeighbors;

class CLocalRegression : public CMapping<double>
{
	protected:
		CKDTree *kdTree;
		CKNearestNeighbors *nearestNeighbors;

		CDataSet *input;
		CDataSet1D *output;

		std::list<int> *subsetList;
		DataSubset *subset;

		int K;

		virtual double doGetOutputValue(ColumnVector *vector);
	
	public:
		CLocalRegression(CDataSet *input, CDataSet1D *output, int K);
		virtual ~CLocalRegression();

		
		virtual double doRegression(ColumnVector *vector, DataSubset *subset) = 0;

		DataSubset *getLastNearestNeighbors();

		int getNumNearestNeighbors();
};

class CLocalRBFRegression : public CLocalRegression
{
	protected:
		ColumnVector *rbfFactors;
		ColumnVector *sigma;
	public:
		CLocalRBFRegression(CDataSet *input, CDataSet1D *output, int K, ColumnVector *sigma);
		virtual ~CLocalRBFRegression();

		virtual double doRegression(ColumnVector *vector, DataSubset *subset);

		ColumnVector *getRBFFactors(ColumnVector *vector,DataSubset *subset);

		ColumnVector *getLastRBFFactors();
};

class CLinearRegression : public CMapping<double>
{
protected:
	Matrix *X;
	ColumnVector *xVector;
	ColumnVector *yVector;
	
	ColumnVector *w;
	
	Matrix *X_pinv;

	int degree;
	
	int numDimensions;
	int xDim;

	void init(int l_degree, int numDataPoints, int numDimensions);

	virtual double doGetOutputValue(ColumnVector *input);
public:
	double lambda;

	CLinearRegression(int degree, int numDataPoints, int numDimensions);
	CLinearRegression(int degree, CDataSet *dataSet, CDataSet1D *outputValues, DataSubset *subset);
	virtual ~CLinearRegression();

	virtual void getXVector(ColumnVector *input, ColumnVector *xVector);
	virtual void calculateRegressionMatrix(CDataSet *dataSet, CDataSet1D *outputValues, DataSubset *subset);

	
};


class CLocalLinearRegression :  public CLocalRegression
{
protected:
	CLinearRegression *regression;

public:
	CLocalLinearRegression(CDataSet *input, CDataSet1D *output, int K, int degree, double lambda);
	~CLocalLinearRegression();

	virtual double doRegression(ColumnVector *vector, DataSubset *subset);
};




#endif
