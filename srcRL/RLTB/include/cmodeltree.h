#ifndef C_MODELTREE__H
#define C_MODELTREE__H

#include "ctrees.h"
#include "clocalregression.h"

class CLinearRegressionDataFactory : public CTreeDataFactory<CMapping<double> *>
{
protected:
	int tresh1;
	int tresh2;
	int tresh3;
	
	CDataSet *input;
	CDataSet1D *output;

	double lambda;
public:
	CLinearRegressionDataFactory(CDataSet *input, CDataSet1D *output, int tresh1, int tresh2, int tresh3, double lambda);
	virtual ~CLinearRegressionDataFactory();

	virtual CMapping<double> *createTreeData(DataSubset *dataSubset, int numLeaves);
	virtual void deleteData(CMapping<double> *linReg);	

};

class CModelTree : public CMapping<double>
{
protected:
	virtual double doGetOutputValue(ColumnVector *input);

	CTree<CMapping<double> *> *tree;

	bool deleteTree;
public:

	CModelTree(CDataSet *inputData, CSplittingConditionFactory *splittingFactory, CTreeDataFactory<CMapping<double> *> *l_dataFactory);
	CModelTree(CTree<CMapping<double> *> *tree); 

	virtual ~CModelTree();

	CTree<CMapping<double> *> *getTree();
};

class CExtraModelTree : public CModelTree
{
public:


	CExtraModelTree(CDataSet *inputData, CDataSet1D *outputData, CTreeDataFactory<CMapping<double> *> *dataFactory, unsigned int K,unsigned  int n_min, double outTresh);

	virtual ~CExtraModelTree();
};


class CExtraLinearRegressionModelTree : public CExtraModelTree
{



public:
	CExtraLinearRegressionModelTree(CDataSet *inputData, CDataSet1D *outputData,  unsigned int K,unsigned  int n_min, double outTresh, int tresh1, int tresh2, int tresh3, double lambda);

	virtual ~CExtraLinearRegressionModelTree();
};


#endif
