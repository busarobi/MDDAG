#ifndef C_KDTREES__H
#define C_KDTREES__H


#include "ctrees.h"

class CDataSet;

class CKDTreeMedianSplittingFactory : public CSplittingConditionFactory
{
protected:
	int n_min;
	CDataSet *inputData;
public:
	CKDTreeMedianSplittingFactory(CDataSet *inputSet, int n_min);
	virtual ~CKDTreeMedianSplittingFactory();

	virtual CSplittingCondition *createSplittingCondition(DataSubset *dataSubset);

	virtual bool isLeaf(DataSubset *dataSubset);
};

class CKDTree : public CTree<DataSubset *>
{
	protected:
		CSplittingConditionFactory *splittingFactory;
	public:
		CKDTree(CDataSet *dataSet, int n_min);
		virtual ~CKDTree();
	
		virtual void addNewInput(int index);
};

#endif
