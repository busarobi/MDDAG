//
// C++ Implementation: crbftrees
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include "crbftrees.h"

#include <math.h>
#include <iostream>
#include "newmat/newmatio.h"

CRBFBasisFunction::CRBFBasisFunction(ColumnVector *l_center, ColumnVector *l_sigma)
{
	center = new ColumnVector(*l_center);
	sigma = new ColumnVector(*l_sigma);
}

double CRBFBasisFunction::getActivationFactor(ColumnVector *x)
{
	double buffer = 0.0;

	for (int i = 0; i < center->nrows(); i ++)
	{
		buffer += pow((x->element(i) - center->element(i)), 2.0) / sigma->element(i);	
	}
	buffer = - buffer / 2;
	/*ColumnVector distance = *x;
	distance = distance - *center;
	printf("Distance To Center: %f\n", distance.norm_Frobenius());*/
	return exp(buffer);
}

ColumnVector *CRBFBasisFunction::getCenter()
{
	return center;
}

ColumnVector *CRBFBasisFunction::getSigma()
{
	return sigma;
}


void CRBFBasisFunction::setSigma(ColumnVector *l_sigma)
{
	sigma = l_sigma;
}

void CRBFBasisFunction::setCenter(ColumnVector *l_center)
{
	center = l_center;
}

CRBFBasisFunctionLinearWeight::CRBFBasisFunctionLinearWeight(ColumnVector *center, ColumnVector *sigma, double l_weight) : CRBFBasisFunction(center, sigma)
{
	weight = l_weight;
}




double CRBFBasisFunctionLinearWeight::getOutputWeight()
{
	return weight;
}

void CRBFBasisFunctionLinearWeight::setWeight(double l_weight)
{
	weight = l_weight;
}

CRBFDataFactory::CRBFDataFactory(CDataSet *l_inputData, ColumnVector *l_varMultiplier, ColumnVector *l_minVar)
{
	minVar = new ColumnVector(*l_minVar);
	varMultiplier = new ColumnVector(*l_varMultiplier);

	inputData = l_inputData;
}

CRBFDataFactory::CRBFDataFactory(CDataSet *l_inputData)
{
	inputData = l_inputData;

	minVar = new ColumnVector(inputData->getNumDimensions());
	varMultiplier = new ColumnVector(inputData->getNumDimensions());

	*minVar = 0.001;
	*varMultiplier = 1.5; 


}

CRBFDataFactory::~CRBFDataFactory()
{
	delete minVar;
	delete varMultiplier;
}

CRBFBasisFunction *CRBFDataFactory::createTreeData(DataSubset *dataSubset, int )
{
	ColumnVector center(inputData->getNumDimensions());
	ColumnVector sigma(inputData->getNumDimensions());

	inputData->getMean( dataSubset, &center);
	inputData->getVariance( dataSubset, &sigma);

	sigma = SP(sigma, *varMultiplier);

	for (int i = 0; i < sigma.nrows(); i ++)
	{
		if (sigma.element(i) < minVar->element(i))
		{
			sigma.element(i) = minVar->element(i);
		}
	}

	return new CRBFBasisFunction(&center, &sigma);
}

void CRBFDataFactory::deleteData(CRBFBasisFunction *basisFunction)
{
	delete basisFunction;
}


CRBFLinearWeightDataFactory::CRBFLinearWeightDataFactory(CDataSet *l_inputData, CDataSet1D *l_outputData, ColumnVector *l_varMultiplier, ColumnVector *l_minVar)
{
	minVar = new ColumnVector(*l_minVar);
	varMultiplier = new ColumnVector(*l_varMultiplier);

	inputData = l_inputData;
	outputData = l_outputData;
}

CRBFLinearWeightDataFactory::~CRBFLinearWeightDataFactory()
{
	delete minVar;
	delete varMultiplier;
}

CRBFBasisFunctionLinearWeight *CRBFLinearWeightDataFactory::createTreeData(DataSubset *dataSubset, int )
{
	ColumnVector center(inputData->getNumDimensions());
	ColumnVector sigma(inputData->getNumDimensions());

	inputData->getMean( dataSubset, &center);
	inputData->getVariance( dataSubset, &sigma);

	sigma = SP(sigma, *varMultiplier);

	double value = 0; //outputData->getMean(dataSubset);
	//double outVar = outputData->getVariance( dataSubset);

	//printf("Creating RBF %d, numSamples %d: center: %f %f , var: %f %f, val: %f, outVar %f\n", numLeaves, dataSubset->size(), center.element(0), center.element(1), sigma.element(0), sigma.element(1), value, outVar);


	for (int i = 0; i < sigma.nrows(); i ++)
	{
		if (sigma.element(i) < minVar->element(i))
		{
			sigma.element(i) = minVar->element(i);
		}
	}

	CRBFBasisFunctionLinearWeight *basisFunction = new CRBFBasisFunctionLinearWeight(&center, &sigma, value);

	DataSubset::iterator it = dataSubset->begin();

	value = 0;
	double norm = 0.0;

	//printf("New Var: %f %f\n", sigma.element(0), sigma.element(1));

	for (; it != dataSubset->end(); it ++)
	{
		double factor = basisFunction->getActivationFactor((*inputData)[*it]);
		norm += factor;
		value += factor * (*outputData)[*it];
		

		//printf("Sample: %f %f (%f %f)\n", factor, (*outputData)[*it], sample->element(0), sample->element(1));
	}
	if (norm > 0)
	{
		value /= norm;
	}
	//printf("value: %f\n", value);
	basisFunction->setWeight( value);
	
	return basisFunction;
}

void CRBFLinearWeightDataFactory::deleteData(CRBFBasisFunctionLinearWeight *basisFunction)
{
	delete basisFunction;
}

CRBFExtraRegressionTree::CRBFExtraRegressionTree(CDataSet *inputData, CDataSet1D *outputData, unsigned int K,unsigned  int n_min, double treshold, ColumnVector *varMultiplier, ColumnVector *minVar) : CExtraTree<CRBFBasisFunctionLinearWeight *>(inputData, outputData, new CRBFLinearWeightDataFactory(inputData, outputData, varMultiplier, minVar), K, n_min, treshold)
{
	
}

CRBFExtraRegressionTree::~CRBFExtraRegressionTree()
{
	delete root;
	root = NULL;
	delete dataFactory;
}


void CKNearestRBFCenters::addDataElements(ColumnVector *point, CLeaf<CRBFBasisFunctionLinearWeight *> *leaf, CKDRectangle *)
{
	CRBFBasisFunctionLinearWeight *basis = leaf->getTreeData();
	
	*buffVector = *basis->getCenter();
	*buffVector = *buffVector - *point;

	addAndSortDataElements(leaf->getLeafNumber(), buffVector->norm_Frobenius());

	/*cout <<"Adding Center (Samples " << leaf->getNumSamples() << "): (" << rectangle->getMinValue(0) << ", " << rectangle->getMaxValue(0) << ") ";
	cout << "(" << rectangle->getMinValue(1) << ", " << rectangle->getMaxValue(1) << ") ";
	
	cout << "(" << rectangle->getMinValue(2) << ", " << rectangle->getMaxValue(2) << ") ";

	cout << basis->getCenter()->t() << endl;*/
}
		
CKNearestRBFCenters::CKNearestRBFCenters(CTree<CRBFBasisFunctionLinearWeight *> *tree, int K) : CKNearestNeighborsTreeData<int, CRBFBasisFunctionLinearWeight *>(tree, K)
{
	buffVector = new ColumnVector(tree->getNumDimensions());
}

CKNearestRBFCenters::~CKNearestRBFCenters()
{
	delete buffVector;
}

CRBFRegressionTreeOutputMapping::CRBFRegressionTreeOutputMapping(CTree<CRBFBasisFunctionLinearWeight *> *l_tree, int K) : CMapping<double>(l_tree->getNumDimensions())
{
	tree = l_tree;

	nearestLeaves = new CKNearestRBFCenters(tree, K);

}

CRBFRegressionTreeOutputMapping::~CRBFRegressionTreeOutputMapping()
{
	delete nearestLeaves;
}

double CRBFRegressionTreeOutputMapping::doGetOutputValue(ColumnVector *input)
{
	std::list<int> neighbors;

	nearestLeaves->getNearestNeighbors(input,  &neighbors);

	double factor = 0.0;
	double value = 0.0;	

	//printf("Activation Factors (%d): ", neighbors.size());
	/*ColumnVector distance = *input;
	ColumnVector mean_distance(input->nrows());
	
	mean_distance = 0;
	
	DataSubset::iterator it = neighbors.begin();
	for (; it != neighbors.end(); it ++)	
	{
		CRBFBasisFunctionLinearWeight *basis = tree->getLeaf(*it)->getTreeData(input);
		
		distance = *input - *basis->getCenter();	
			
		mean_distance = distance + mean_distance;
	}
	mean_distance = mean_distance / neighbors.size() * 2;
	*/

	std::list<int>::iterator it = neighbors.begin();
	for (; it != neighbors.end(); it ++)	
	{
		CRBFBasisFunctionLinearWeight *basis = tree->getLeaf(*it)->getTreeData();
		
		//basis->setSigma(&mean_distance);

		double l_factor = basis->getActivationFactor(input);

		factor += l_factor;
		value += l_factor * basis->getOutputWeight();

		//printf("(%f %f %f %f) ", l_factor, basis->getOutputWeight(), factor, value);
	}

	//cout << "Input: " << input->t() << endl;
	it = neighbors.begin();	
	for (; it != neighbors.end(); it ++)	
	{
		CRBFBasisFunctionLinearWeight *basis = tree->getLeaf(*it)->getTreeData();
		
//		ColumnVector dist(*basis->getCenter());
//		cout << "Center: " << dist.t() << " ";
//		dist = dist - *input;
		double l_factor = basis->getActivationFactor(input);
	
		factor += l_factor;
		value += l_factor * basis->getOutputWeight();
			

	//	printf("(%f %f %f %f)\n", l_factor, basis->getOutputWeight(), dist.norm_Frobenius(), basis->getSigma()->norm_Frobenius());
	}
	//printf("\n");

	if (factor > 0)
	{
		value = value / factor;
	}
	else
	{
		printf("Warning: RBF Tree network: Summed Factor == %f (%d)!!!\n", factor, neighbors.size());

		it = neighbors.begin();	
		cout << "Input : " << input->t() << endl;
		for (; it != neighbors.end(); it ++)	
		{
			CRBFBasisFunctionLinearWeight *basis = tree->getLeaf(*it)->getTreeData();
		
			ColumnVector dist(*basis->getCenter());
			cout << "Center (Samples " << tree->getLeaf(*it)->getNumSamples() << "): " << dist.t() << " ";
			dist = dist - *input;
		
			double l_factor = basis->getActivationFactor(input);
	
			factor += l_factor;
			value += l_factor * basis->getOutputWeight();
			

			printf("(%f %f %f %f)\n", l_factor, basis->getOutputWeight(), dist.norm_Frobenius(), basis->getSigma()->norm_Frobenius());
		}

		std::vector<double> distances;
		CDataSet *inputData = tree->getInputData();
		for (unsigned int j = 0; j < inputData->size(); j ++)
		{
			ColumnVector distance(*(*inputData)[j]);
			distance = distance - *input;
			double dist = distance.norm_Frobenius();	

			distances.push_back(dist);
		}
		std::sort(distances.begin(), distances.end());

		printf("Distances of NNs (Real) : ");
		for (unsigned int j = 0; j < 5 && j < distances.size(); j ++)
		{
			printf("%f ", distances[j]);
		}
		printf("\n");
		//assert(false);
	}
	
	return value;
}


CRBFExtraRegressionForest::CRBFExtraRegressionForest(int numTrees, int kNN, CDataSet *inputData, CDataSet1D *outputData, unsigned int K,unsigned  int n_min, double treshold, ColumnVector *varMultiplier, ColumnVector *minVar) : CForest<CRBFBasisFunctionLinearWeight *>(numTrees), CMapping<double>(inputData->getNumDimensions())
{
	dataFactory = new CRBFLinearWeightDataFactory(inputData, outputData, varMultiplier, minVar);
	for (int i = 0; i < numTrees; i++)
	{
		addTree( i, new CExtraTree<CRBFBasisFunctionLinearWeight *>(inputData, outputData, dataFactory, K, n_min, treshold));		
	}
	mapping = new CRBFRegressionTreeOutputMapping *[numTrees];
	initRBFMapping(kNN);
}

CRBFExtraRegressionForest::~CRBFExtraRegressionForest()
{
	for (int i = 0; i < numTrees; i++)
	{
		delete forest[i];
		delete mapping[i];
	}
	delete dataFactory;
}

void CRBFExtraRegressionForest::initRBFMapping(int kNN)
{
	for (int i = 0; i < numTrees;i ++)
	{
		mapping[i] = new CRBFRegressionTreeOutputMapping(forest[i], kNN);
	}
}

double CRBFExtraRegressionForest::doGetOutputValue(ColumnVector *input)
{
	double mean = 0;	

	for (int i = 0; i < numTrees; i++)
	{
		mean += mapping[i]->getOutputValue(input);
	}
	return mean / numTrees;
};



CRBFLinearWeightForest::CRBFLinearWeightForest(int numTrees, int numDim) : CForest<CRBFBasisFunctionLinearWeight *>(numTrees) , CMapping<double>(numDim)
{

}

CRBFLinearWeightForest::~CRBFLinearWeightForest()
{
}

void CRBFLinearWeightForest::saveASCII(FILE *stream)
{
	fprintf(stream, "%f %f\n", getAverageDepth(), getAverageNumLeaves());
}

double CRBFLinearWeightForest::getOutputValue(ColumnVector *input)
{
	double sum = 0;
	double norm = 0;
	for (int i = 0; i < numTrees; i ++)
	{
		CTree<CRBFBasisFunctionLinearWeight *> *tree = getTree( i);
		CRBFBasisFunctionLinearWeight *rbfBasis = tree->getOutputValue(input);
		double factor = rbfBasis->getActivationFactor(input);
		//printf("(%f, %f) ", factor, rbfBasis->getOutputWeight());
		sum += factor * rbfBasis->getOutputWeight();
		norm += factor;
	}
	//printf("\n");
	if (fabs(norm) > 0)
	{
		sum = sum / norm;
	}
	return sum;
}


CExtraTreeRBFLinearWeightForest::CExtraTreeRBFLinearWeightForest(int numTrees, CDataSet *inputData, CDataSet1D *outputData, unsigned int K,unsigned  int n_min, double treshold, ColumnVector *varMultiplier, ColumnVector *minVar) : CRBFLinearWeightForest(numTrees, inputData->getNumDimensions())
{
	dataFactory = new CRBFLinearWeightDataFactory(inputData, outputData, varMultiplier, minVar);
	for (int i = 0; i < numTrees; i++)
	{
		addTree( i, new CExtraTree<CRBFBasisFunctionLinearWeight *>(inputData, outputData, dataFactory, K, n_min, treshold));		

	}
}

CExtraTreeRBFLinearWeightForest::~CExtraTreeRBFLinearWeightForest()
{
	for (int i = 0; i < numTrees; i++)
	{
		delete forest[i];
	}
	delete dataFactory;
}

