#include "cmodeltree.h"

#include "cdatafactory.h"
#include "cextratrees.h"


CLinearRegressionDataFactory::CLinearRegressionDataFactory(CDataSet *l_input, CDataSet1D *l_output, int l_tresh1, int l_tresh2, int l_tresh3, double l_lambda)
{
	input = l_input;
	output = l_output;
	
	tresh1 = l_tresh1;
	tresh2 = l_tresh2;
	tresh3 = l_tresh3;

	lambda = l_lambda;
}


CLinearRegressionDataFactory::~CLinearRegressionDataFactory()
{
}

CMapping<double> *CLinearRegressionDataFactory::createTreeData(DataSubset *dataSubset, int )
{
	int degree = 0;
	if (dataSubset->size() < (unsigned int) tresh1)
	{
		degree = 0;
	}
	else
	{
		if (dataSubset->size() < (unsigned int) tresh2)
		{
			degree = 1;
		}
		else
		{
			if (dataSubset->size() < (unsigned int) tresh3)
			{
				degree = 2;
			}
			else
			{
				degree = 3;
			}
		}
	}

  	CLinearRegression *regression = new CLinearRegression(degree, input, output, dataSubset);

	regression->lambda = lambda;

	return regression;
}
	
void CLinearRegressionDataFactory::deleteData(CMapping<double> *linReg)
{
	delete linReg;
}

double CModelTree::doGetOutputValue(ColumnVector *input)
{
	CMapping<double> *linReg = tree->getOutputValue(input);

	return linReg->getOutputValue(input);
}

CModelTree::CModelTree(CDataSet *inputData, CSplittingConditionFactory *splittingFactory, CTreeDataFactory<CMapping<double> *> *l_dataFactory) : CMapping<double>(inputData->getNumDimensions())
{
	tree = new CTree<CMapping<double> *>(inputData, splittingFactory, l_dataFactory);
	deleteTree = true;
}

CModelTree::CModelTree(CTree<CMapping<double> *> *l_tree) : CMapping<double>(l_tree->getNumDimensions())
{
	tree = l_tree;
	deleteTree = false;
}

CModelTree::~CModelTree()
{
	if (deleteTree && tree)
	{	
		delete tree;
	}
	
}

CTree<CMapping<double> *> *CModelTree::getTree()
{
	return tree;
}

CExtraModelTree::CExtraModelTree(CDataSet *inputData, CDataSet1D *outputData, CTreeDataFactory<CMapping<double> *> *dataFactory, unsigned int K,unsigned  int n_min, double outTresh) : CModelTree(new CExtraTree<CMapping<double> *>(inputData, outputData, dataFactory, K, n_min, outTresh))
{
}

CExtraModelTree::~CExtraModelTree()
{
	if (tree)
	{
		delete tree;
	}
}

CExtraLinearRegressionModelTree::CExtraLinearRegressionModelTree(CDataSet *inputData, CDataSet1D *outputData,  unsigned int K,unsigned  int n_min, double outTresh, int tresh1, int tresh2, int tresh3, double lambda) : CExtraModelTree(inputData, outputData, new CLinearRegressionDataFactory(inputData, outputData, tresh1, tresh2, tresh3, lambda), K, n_min, outTresh)
{
	
}

CExtraLinearRegressionModelTree::~CExtraLinearRegressionModelTree()
{
	if (tree)
	{
		CTreeDataFactory<CMapping<double> *> *dataFactory = tree->getDataFactory();
		delete tree;
		delete dataFactory;
		tree = NULL;
	}

	
}

