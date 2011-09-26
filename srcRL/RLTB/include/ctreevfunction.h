#ifndef C_TREEVFUNCTION__H
#define C_TREEVFUNCTION__H

#include "ctrees.h"
#include "cforest.h"
#include "cinputdata.h"
#include "cvfunction.h"
#include "ccontinuousactions.h"
#include "cstatemodifier.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "caction.h"


class CRegressionTreeFunction
{
protected:
	CMapping<double> *tree;
	
	int numDim;
public:
	CRegressionTreeFunction(CMapping<double> *tree, int numDim);

	virtual ~CRegressionTreeFunction() {};
	void setTree(CMapping<double> *tree);
	CMapping<double> *getTree();

	int getNumDimensions();
	virtual void getInputData(CStateCollection *state, CAction *action, ColumnVector *data) = 0;
};

class CRegressionTreeVFunction : public CAbstractVFunction, public CRegressionTreeFunction
{
	protected :
	
	public :
		CRegressionTreeVFunction(CStateProperties *properties, CMapping<double> *tree);

		virtual ~CRegressionTreeVFunction() {};

		virtual double getValue(CState *state);
		virtual void getInputData(CStateCollection *state, CAction *action, ColumnVector *data);

		virtual void resetData();
		virtual void saveData(FILE *stream);
};

class CRegressionTreeQFunction : public CContinuousActionQFunction, public CStateObject, public CRegressionTreeFunction
{
	protected :
	
		ColumnVector *buffVector;
	public :
		CRegressionTreeQFunction(CContinuousAction *action, CStateProperties *properties, CMapping<double> *tree);

		virtual ~CRegressionTreeQFunction();

		virtual double getCAValue(CStateCollection *state, CContinuousActionData *data);

		virtual void getInputData(CStateCollection *state, CAction *action, ColumnVector *data);

		virtual void resetData();
};

template<typename TreeData> class CForestFeatureCalculator : public CFeatureCalculator
{
protected:
	CForest<TreeData> *forest;
	CLeaf<TreeData> **activeLeaves;

	double  getLeafActivationFactor(CState *stateCol, CLeaf<TreeData> *targetState);
public:
	CForestFeatureCalculator(CForest<TreeData> *forest, int offsetNumLeaves = 0);
	
	CForestFeatureCalculator(int numFeatures, int numActiveFeatures);
	

	virtual ~CForestFeatureCalculator();

	void getModifiedState(CStateCollection *stateCol, CState *targetState);

	void setForest(CForest<TreeData> *forest);
};

template<typename TreeData> CForestFeatureCalculator<TreeData>::CForestFeatureCalculator(CForest<TreeData> *l_forest, int offsetNumTrees) : CFeatureCalculator(l_forest->getNumLeaves() + offsetNumTrees, l_forest->getNumTrees())
{
	forest = l_forest;
	activeLeaves = new CLeaf<TreeData>*[forest->getNumTrees()];
}

template<typename TreeData> CForestFeatureCalculator<TreeData>::CForestFeatureCalculator(int numFeatures, int numActiveFeatures) : CFeatureCalculator(numFeatures, numActiveFeatures)
{
	forest = NULL;
	activeLeaves = new CLeaf<TreeData>*[getNumActiveFeatures()];
}

template<typename TreeData> CForestFeatureCalculator<TreeData>::~CForestFeatureCalculator()
{
	delete [] activeLeaves;

	if (forest)
	{
		delete forest;
	}
}


template<typename TreeData> void CForestFeatureCalculator<TreeData>::setForest(CForest<TreeData> *l_forest)
{
	forest = l_forest;	
}

template<typename TreeData> void  CForestFeatureCalculator<TreeData>::getModifiedState(CStateCollection *stateCol, CState *targetState)
{
	if (forest == NULL)
	{
		targetState->resetState();
		targetState->setNumActiveContinuousStates(0);
		targetState->setNumActiveDiscreteStates(0);
	}
	else
	{	
		CState *state = stateCol->getState(originalState);
	
		forest->getActiveLeaves( state, activeLeaves);
	
		int leafSum = 0;
	
		targetState->setNumActiveContinuousStates(numActiveFeatures);
		targetState->setNumActiveDiscreteStates(numActiveFeatures);
		
		for (unsigned int i = 0; i < numActiveFeatures; i ++)
		{
			targetState->setDiscreteState(i, activeLeaves[i]->getLeafNumber() + leafSum);
			targetState->setContinuousState(i, getLeafActivationFactor(state, activeLeaves[i]));
	
			leafSum += forest->getTree( i)->getNumLeaves();
		}
	}
}

template<typename TreeData> double  CForestFeatureCalculator<TreeData>::getLeafActivationFactor(CState *, CLeaf<TreeData> *)
{
	return 1.0 / numActiveFeatures;
}

class CFeatureVRegressionTreeFunction : public CFeatureVFunction
{
protected:

public:
	CFeatureVRegressionTreeFunction(CRegressionForest *regTree, CFeatureCalculator *featCalc);

	CFeatureVRegressionTreeFunction(int numFeatures);

	void setForest(CRegressionForest *regTree, CFeatureCalculator *featCalc);
	
	virtual double getValue(CState *state);

	virtual void copy(CLearnDataObject *vFunction);
};

#endif
