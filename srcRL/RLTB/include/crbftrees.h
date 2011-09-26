//
// C++ Interface: crbftrees
//
// Description: 
//
//
// Author: Neumann Gerhard <gerhard@tu-graz.ac.at>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef C_RBFTREES__H
#define C_RBFTREES__H



#include "cinputdata.h"
#include "ctrees.h"
#include "cforest.h"
#include "cextratrees.h"
#include "cnearestneighbor.h"

#include <newmat/newmat.h>
#include <list>
#include <vector>
#include <stdio.h>

class CRBFBasisFunction 
{
	protected:
		ColumnVector *center;
		ColumnVector *sigma;
	public:
		CRBFBasisFunction(ColumnVector *center, ColumnVector *sigma);


		double getActivationFactor(ColumnVector *x);
		
		ColumnVector *getCenter();
		ColumnVector *getSigma();

		void setSigma(ColumnVector *sigma);
		void setCenter(ColumnVector *center);
};

class CRBFBasisFunctionLinearWeight : public CRBFBasisFunction 
{
	protected:
		double weight;
	public:
		CRBFBasisFunctionLinearWeight(ColumnVector *center, ColumnVector *sigma, double weight);


		double getOutputWeight();
		void setWeight(double weight);
};

class CRBFDataFactory : public CTreeDataFactory<CRBFBasisFunction *>
{
protected:
	ColumnVector *minVar;
	ColumnVector *varMultiplier;

	CDataSet *inputData;
public:
	CRBFDataFactory(CDataSet *inputData, ColumnVector *varMultiplier, ColumnVector *minVar);
	CRBFDataFactory(CDataSet *inputData);
	virtual ~CRBFDataFactory();


	virtual CRBFBasisFunction *createTreeData(DataSubset *dataSubset, int numLeaves);
	virtual void deleteData(CRBFBasisFunction *basisFunction); 
};

class CRBFLinearWeightDataFactory : public CTreeDataFactory<CRBFBasisFunctionLinearWeight *>
{
protected:
	ColumnVector *minVar;
	ColumnVector *varMultiplier;

	CDataSet *inputData;
	CDataSet1D *outputData;
public:
	CRBFLinearWeightDataFactory(CDataSet *inputData, CDataSet1D *outputData, ColumnVector *varMultiplier, ColumnVector *minVar);
	virtual ~CRBFLinearWeightDataFactory();


	virtual CRBFBasisFunctionLinearWeight *createTreeData(DataSubset *dataSubset, int numLeaves);
	virtual void deleteData(CRBFBasisFunctionLinearWeight *basisFunction); 
};

class CRBFExtraRegressionTree : public CExtraTree<CRBFBasisFunctionLinearWeight *>
{
	public:
		CRBFExtraRegressionTree(CDataSet *inputData, CDataSet1D *outputData, unsigned int K,unsigned  int n_min, double treshold, ColumnVector *varMultiplier, ColumnVector *minVar);
		virtual ~CRBFExtraRegressionTree();
};

class CKNearestRBFCenters : public  CKNearestNeighborsTreeData<int, CRBFBasisFunctionLinearWeight *>
{
	protected:
		ColumnVector *buffVector;

		virtual void addDataElements(ColumnVector *point, CLeaf<CRBFBasisFunctionLinearWeight *> *leaf, CKDRectangle *rectangle);
		
	public:
		CKNearestRBFCenters(CTree<CRBFBasisFunctionLinearWeight *> *tree, int K);
		virtual ~CKNearestRBFCenters();
};

class CRBFRegressionTreeOutputMapping : public CMapping<double>
{
	protected:
		CTree<CRBFBasisFunctionLinearWeight *> *tree;
		CKNearestRBFCenters *nearestLeaves;

		double doGetOutputValue(ColumnVector *output);
	public:
		CRBFRegressionTreeOutputMapping(CTree<CRBFBasisFunctionLinearWeight *> *tree, int K);
		virtual ~CRBFRegressionTreeOutputMapping();

		
};

class CRBFExtraRegressionForest : public CForest<CRBFBasisFunctionLinearWeight *>, public CMapping<double>
{
protected:
	CRBFRegressionTreeOutputMapping **mapping;
	CTreeDataFactory<CRBFBasisFunctionLinearWeight *> *dataFactory;

	void initRBFMapping(int kNN);

	virtual double doGetOutputValue(ColumnVector *input);
public:
	CRBFExtraRegressionForest(int numTrees, int kNN, CDataSet *inputData, CDataSet1D *outputData, unsigned int K,unsigned  int n_min, double treshold, ColumnVector *varMultiplier, ColumnVector *minVar);
	virtual ~CRBFExtraRegressionForest();

	
};

class CRBFLinearWeightForest : public CForest<CRBFBasisFunctionLinearWeight *>, public CMapping<double>
{
	public:

	protected:
		CRBFLinearWeightForest(int numTrees, int numDim);
		virtual ~CRBFLinearWeightForest();

		double getOutputValue(ColumnVector *inputData);

		virtual void saveASCII(FILE *stream);

};


class CExtraTreeRBFLinearWeightForest : public CRBFLinearWeightForest
{
	protected:
		CRBFLinearWeightDataFactory *dataFactory;

	public:

		CExtraTreeRBFLinearWeightForest(int numTrees, CDataSet *inputData, CDataSet1D *outputData, unsigned int K,unsigned  int n_min, double treshold, ColumnVector *varMultiplier, ColumnVector *minVar);
		virtual ~CExtraTreeRBFLinearWeightForest();
};


#endif
