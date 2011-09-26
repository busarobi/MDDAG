#include "ctreevfunction.h"
#include "cstate.h"
#include "cstatecollection.h"


CRegressionTreeFunction::CRegressionTreeFunction(CMapping<double> *l_tree, int l_numDim)
{
	tree = l_tree;
	numDim = l_numDim;
}

void CRegressionTreeFunction::setTree(CMapping<double> *l_tree)
{
	tree = l_tree;
//	assert(tree->getNumDimensions() == vector->nrows());
}

CMapping<double> *CRegressionTreeFunction::getTree()
{
	return tree;
}

int CRegressionTreeFunction::getNumDimensions()
{
	return numDim;
}

CRegressionTreeVFunction::CRegressionTreeVFunction(CStateProperties *properties, CMapping<double> *l_tree) : CAbstractVFunction(properties), CRegressionTreeFunction(l_tree, properties->getNumContinuousStates())
{
	

//	assert(tree->getNumDimensions() == properties->getNumContinuousStates());
}

double CRegressionTreeVFunction::getValue(CState *state)
{
	if (tree == NULL)
	{
		return 0;
	}
	double value = tree->getOutputValue(state);
	
	return value;
}
 
void CRegressionTreeVFunction::resetData()
{
	if (tree != NULL)
	{
		delete tree;	
	}
	tree = NULL;
}

void CRegressionTreeVFunction::saveData(FILE *stream)
{
	if (tree)
	{
		tree->saveASCII(stream);
	}
}

void CRegressionTreeVFunction::getInputData(CStateCollection *stateCol, CAction *, ColumnVector *buffVector)
{
	CState *state = stateCol->getState(properties);
	
	*buffVector = *state;
}

CRegressionTreeQFunction::CRegressionTreeQFunction(CContinuousAction *action, CStateProperties *properties, CMapping<double> *l_tree) : CContinuousActionQFunction(action), CStateObject(properties), CRegressionTreeFunction(l_tree, properties->getNumContinuousStates() + action->getNumDimensions())
{
	
	buffVector = new ColumnVector(properties->getNumContinuousStates() + action->getNumDimensions());

//	assert(tree->getNumDimensions() == buffVector->nrows());
}

CRegressionTreeQFunction::~CRegressionTreeQFunction()
{
	delete buffVector;
}

double CRegressionTreeQFunction::getCAValue(CStateCollection *stateCol, CContinuousActionData *data)
{
	if (tree == NULL)
	{
		return 0;
	}

	CState *state = stateCol->getState( properties);
	for (unsigned int i = 0; i < properties->getNumContinuousStates(); i ++)
	{
		buffVector->element(i) = state->getContinuousState( i);
	}
	int dim =  properties->getNumContinuousStates();
	for (int i = 0; i < data->nrows(); i ++)
	{
		buffVector->element(i + dim) = data->element(i);
	}

	return tree->getOutputValue(buffVector);
}

void CRegressionTreeQFunction::getInputData(CStateCollection *stateCol, CAction *action, ColumnVector *buffVector)
{
	CContinuousActionData *data = dynamic_cast<CContinuousAction *>(action)->getContinuousActionData();	

	CState *state = stateCol->getState( properties);
	for (unsigned int i = 0; i < properties->getNumContinuousStates(); i ++)
	{
		buffVector->element(i) = state->getContinuousState( i);
	}
	int dim =  properties->getNumContinuousStates();
	for (int i = 0; i < data->nrows(); i ++)
	{
		buffVector->element(i + dim) = data->element(i);
	}
}


void CRegressionTreeQFunction::resetData()
{
	if (tree != NULL)
	{
		delete tree;	
	}
	tree = NULL;
}

CFeatureVRegressionTreeFunction::CFeatureVRegressionTreeFunction(CRegressionForest *regForest, CFeatureCalculator *featCalc) : CFeatureVFunction(featCalc)
{
	setForest(regForest, featCalc);
}

CFeatureVRegressionTreeFunction::CFeatureVRegressionTreeFunction(int numFeatures) : CFeatureVFunction(numFeatures)
{
	
}

void CFeatureVRegressionTreeFunction::setForest(CRegressionForest *regForest, CFeatureCalculator *featCalc)
{

	if (numFeatures < featCalc->getNumFeatures())
	{	
		printf("Setting Forest with too many features: V-Function: %d Forest: %d\n", numFeatures, featCalc->getNumFeatures());
		assert(numFeatures >= featCalc->getNumFeatures());
	}
	properties = featCalc;

	int feature = 0;
	resetData();
	for (int i = 0; i < regForest->getNumTrees(); i ++)
	{
		CTree<double> *regTree = regForest->getTree(i);
		for (int j = 0; j < regTree->getNumLeaves(); j ++)
		{
			setFeature( feature, regTree->getLeaf( j)->getTreeData());
			feature ++;
		}
	}
}

double CFeatureVRegressionTreeFunction::getValue(CState *state)
{
	if (properties == NULL)
	{
		return 0.0;
	}
	else
	{
		return CFeatureVFunction::getValue(state);
	}
}

void CFeatureVRegressionTreeFunction::copy(CLearnDataObject *vFunction)
{
	CFeatureVRegressionTreeFunction *regVFunction = dynamic_cast<CFeatureVRegressionTreeFunction *>(vFunction);

	CFeatureVFunction::copy(vFunction);

	regVFunction->setFeatureCalculator(dynamic_cast<CFeatureCalculator *>(properties));	
}
