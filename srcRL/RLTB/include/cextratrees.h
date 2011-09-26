#ifndef C_EXTRA_TREES__H
#define C_EXTRA_TREES__H

#include "ctrees.h"

#include <newmat/newmat.h>

class CDataSet;
class CDataSet1D;

class CExtraTreesSplittingConditionFactory : public CSplittingConditionFactory
{
	protected:
		unsigned int K;
		unsigned int n_min;
		
		double outTreshold;
				
		CDataSet *inputData;
		CDataSet1D *outputData;
		CDataSet1D *weightingData;
				
		double getScore(CSplittingCondition *condition, DataSubset *dataSubset);
	public:
		CExtraTreesSplittingConditionFactory(CDataSet *inputData, CDataSet1D *outputData, unsigned int K, unsigned int n_min, double outTresh = 0.0, CDataSet1D *weightingData = NULL);
		virtual ~CExtraTreesSplittingConditionFactory();
		
		virtual CSplittingCondition *createSplittingCondition(DataSubset *dataSubset);

		virtual bool isLeaf(DataSubset *dataSubset);
};

template <typename TreeData> class CExtraTree : public CTree<TreeData>
{
	public:
		CExtraTree(CDataSet *inputData, CDataSet1D *outputData, CTreeDataFactory<TreeData> *dataFactory, unsigned int K,unsigned  int n_min, double outTresh, CDataSet1D *weightingData = NULL) : CTree<TreeData>(inputData->getNumDimensions())
		{
			CSplittingConditionFactory *splittingFactory = new CExtraTreesSplittingConditionFactory(inputData, outputData, K, n_min, outTresh, weightingData);
			CTree<TreeData>::createTree(inputData, splittingFactory, dataFactory);
			delete splittingFactory;
		};
		
		virtual ~CExtraTree()
		{
			
		};
};

class CExtraRegressionTree : public CExtraTree<double>
{
	public:
		CExtraRegressionTree(CDataSet *inputData, CDataSet1D *outputData, unsigned int K,unsigned  int n_min, double treshold, CDataSet1D *weightingData = NULL);
		virtual ~CExtraRegressionTree();
};





#endif
