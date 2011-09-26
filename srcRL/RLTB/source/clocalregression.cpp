#include "clocalregression.h"
 
#include <math.h>
#include "cutility.h"


#include "ctrees.h"
#include "ckdtrees.h"
#include "cnearestneighbor.h"

CLocalRegression::CLocalRegression(CDataSet *l_input, CDataSet1D *l_output, int l_K) : CMapping<double>(l_input->getNumDimensions())
{
	input = l_input;
	output = l_output;
		
	printf("Building Kd-Tree with %d inputs\n", input->size());
	kdTree = new CKDTree(input, 1);

	printf("Tree: %d %d\n", kdTree->getDepth(), kdTree->getNumLeaves());
	
	nearestNeighbors = new CKNearestNeighbors(kdTree, input, l_K);

	subsetList = new std::list<int>();
	subset = new DataSubset();

	K = l_K;
	
	if (buffVector->size() == 0)
	{
		printf("%d %d %d\n", buffVector->size(), l_input->getNumDimensions(), l_input->size());
		assert(buffVector->size() > 0);
	}
}
	
CLocalRegression::~CLocalRegression()
{
	delete kdTree;
	delete nearestNeighbors;

	delete subset;
	delete subsetList;
}

double CLocalRegression::doGetOutputValue(ColumnVector *vector)
{
	subset->clear();
	subsetList->clear();	
	
	nearestNeighbors->getNearestNeighbors( vector, subsetList);

	subset->addElements(subsetList);

	return doRegression(vector, subset);
}
	
DataSubset *CLocalRegression::getLastNearestNeighbors()
{
	return subset;
}

int CLocalRegression::getNumNearestNeighbors()
{
	return K;
}


CLinearRegression::CLinearRegression(int degree, int numDataPoints, int numDimensions) : CMapping<double>(numDimensions)
{
	init(degree, numDataPoints, numDimensions);	
}

CLinearRegression::CLinearRegression(int degree, CDataSet *dataSet, CDataSet1D *outputValues, DataSubset *subset) : CMapping<double>(dataSet->getNumDimensions())
{
	if (subset)
	{
		init(degree, subset->size(), dataSet->getNumDimensions());
	}
	else
	{
		init(degree, dataSet->size(), dataSet->getNumDimensions());
	}
	calculateRegressionMatrix(dataSet, outputValues, subset);
}


void CLinearRegression::init(int l_degree, int numDataPoints, int l_numDimensions)
{
	degree = l_degree;
	numDimensions = l_numDimensions;

	switch (degree)
	{
		case 0:
		{
			xDim = 1;
			break;
		}
		case 1:
		{
			xDim = 1 + numDimensions;
			break;
		}
		case 2:
		{
			xDim = 1 + numDimensions * numDimensions;
			break;
		}
		case 3:
		{
			xDim = 1 + numDimensions * (numDimensions + 1);
			break;
		}
	};

	X = new Matrix(numDataPoints, xDim);
	X_pinv = new Matrix(xDim, numDataPoints);	

	xVector = new ColumnVector(xDim);
	yVector = new ColumnVector(numDataPoints);

	w = new ColumnVector(xDim);
	
	lambda = 0.01;
}

CLinearRegression::~CLinearRegression()
{
	delete X;
	delete xVector;
	delete yVector;

	delete w;
	delete X_pinv;
}

void CLinearRegression::getXVector(ColumnVector *input, ColumnVector *xVector)
{
	xVector->element(0) = 1;

	// Linear part
	if (degree >= 1)
	{
		for (int i = 0; i < numDimensions; i ++)
		{
			xVector->element(i + 1) =  input->element(i);
		}
	}

	// Quadratic Part
	if (degree >= 2)
	{
		int count = 0; 
		
		for (int i = 0; i < numDimensions; i ++)
		{
			for (int j = 0; j < numDimensions; j ++)
			{
				if (degree > 2 || i != j)
				{
					xVector->element(1 + numDimensions + count) = input->element(i) * input->element(j);
					count ++;
				}
			}
		}
	}
}

void CLinearRegression::calculateRegressionMatrix(CDataSet *dataSet, CDataSet1D *outputValues, DataSubset *subset)
{
	if (subset)
	{
		assert((signed int) subset->size() == X->nrows());
		DataSubset::iterator it = subset->begin();
		for (int i = 0; it != subset->end(); it ++, i++)
		{
			yVector->element(i) = (*outputValues)[*it];

			getXVector((*dataSet)[*it], xVector);

			for (int j = 0; j < xVector->size(); j ++)
			{
				X->element(i, j) = xVector->element(j);	
			}
		}
	}
	else
	{
		assert(dataSet->size() == (unsigned int) X->nrows());
		
		for (unsigned int i = 0; i < dataSet->size();  i++)
		{
			yVector->element(i) = (*outputValues)[i];

			getXVector((*dataSet)[i], xVector);

			for (int j = 0; j < xVector->size(); j ++)
			{
				X->element(i, j) = xVector->element(j);	
			}
		}
	
	}

	//printf("X: %d %d  X_pinv: %d %d\n", X->nrows(), X->ncols(), X_pinv->nrows(), X_pinv->ncols());
	getPseudoInverse(X, X_pinv, lambda);

	*w = (*X_pinv) * (*yVector);
}


double CLinearRegression::doGetOutputValue(ColumnVector *input)
{
	getXVector(input, xVector);

	return dotproduct(*w, *xVector);
}


CLocalLinearRegression::CLocalLinearRegression(CDataSet *input, CDataSet1D *output, int K, int degree, double lambda) : CLocalRegression(input, output, K)
{
	regression = new CLinearRegression(degree, K, input->getNumDimensions());

	regression->lambda = lambda;
}

CLocalLinearRegression::~CLocalLinearRegression()
{
	delete regression;
}

double CLocalLinearRegression::doRegression(ColumnVector *vector, DataSubset *subset)
{
	regression->calculateRegressionMatrix(input, output, subset);
	
	return regression->getOutputValue(vector);
}



CLocalRBFRegression::CLocalRBFRegression(CDataSet *input, CDataSet1D *output, int K, ColumnVector *l_sigma) : CLocalRegression(input, output, K)
{
	rbfFactors = new ColumnVector(K);
	sigma = new ColumnVector(*l_sigma);
}

CLocalRBFRegression::~CLocalRBFRegression()
{
	delete sigma;
	delete rbfFactors;
}

double CLocalRBFRegression::doRegression(ColumnVector *vector, DataSubset *subset)
{
//	cout << "NNs for Input " << vector->t() << endl;
	
	DataSubset::iterator it = subset->begin();
/*	for (int i = 0; it != subset->end(); it ++, i++)
	{
		ColumnVector dist(*vector);
		dist = dist - *(*input)[*it];
		printf("(%d %f) ", *it, dist.norm_Frobenius());
	}
	printf("\n"); */

	ColumnVector *rbfFactors = getRBFFactors(vector, subset);
	it = subset->begin();
	double value = 0;
	for (int i = 0; it != subset->end(); it ++, i++)
	{
		value += rbfFactors->element(i) * (*output)[*it];
	}
	double sum = rbfFactors->sum();
	if (sum > 0)
	{
		value = value / sum;
	}
	//printf("Value: %f %f ", value ,sum);
	//cout << rbfFactors->t();
	return value;
}

ColumnVector *CLocalRBFRegression::getRBFFactors(ColumnVector *vector, DataSubset *subset)
{
	DataSubset::iterator it = subset->begin();

	for (int i = 0; it != subset->end(); it ++, i ++)
	{
		double malDist = 0;
		for (int j = 0; j < vector->nrows(); j ++)
		{
			ColumnVector *data = (*input)[*it];
			malDist += pow((vector->element(j) - data->element(j)) / sigma->element(j), 2.0);
		}

		rbfFactors->element(i) = exp(- malDist / 2.0);
	}
//	cout << "RBFFactors: " << rbfFactors->t() << endl;

	return rbfFactors;
}

ColumnVector *CLocalRBFRegression::getLastRBFFactors()
{
	return rbfFactors;
}



