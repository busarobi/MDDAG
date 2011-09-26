#include "clstd.h"

#include <newmat/newmat.h>

#include "cvfunction.h"
#include "cvetraces.h"

#include "cvfunction.h"
#include "cstateproperties.h"
#include "cstatemodifier.h"
#include "cstate.h"
#include "cstatecollection.h"

#include "caction.h"
#include "cagentcontroller.h"
#include "cqetraces.h"
#include "cqfunction.h"
#include "crewardfunction.h"

CLSTDLambda::CLSTDLambda(CRewardFunction *rewardFunction, CGradientUpdateFunction *l_vFunction, int l_nUpdatePerEpisode) : CSemiMDPRewardListener(rewardFunction), CLeastSquaresLearner(l_vFunction, l_vFunction->getNumWeights())
{
	//vFunction = l_vFunction;
	nEpisode = 1;
	nUpdateEpisode = l_nUpdatePerEpisode;
	
	//featureCalc = vFunction->getStateProperties();
	
	newStateGradient = new CFeatureList();
	oldStateGradient = new CFeatureList();
}

CLSTDLambda::~CLSTDLambda()
{
	
	delete newStateGradient; 
	delete oldStateGradient;
}
	
void CLSTDLambda::nextStep(CStateCollection *oldStateCol, CAction *action, double reward, CStateCollection *newStateCol)
{
	//vETraces->updateETraces(action->getDuration());
	//vETraces->addETrace(oldStateCol);
	double gamma = getParameter("DiscountFactor");
	
	updateETraces(oldStateCol, action);
	
	oldStateGradient->clear();
	newStateGradient->clear();
	
	if (! newStateCol->getState()->isResetState())
	{
		getNewGradient(newStateCol, newStateGradient);
		newStateGradient->multFactor(- gamma);
	}
	getOldGradient(oldStateCol, action, oldStateGradient);
	newStateGradient->add(oldStateGradient);
			
	CFeatureList *eTraceList = getGradientETraces();
	
	CFeatureList::iterator it = newStateGradient->begin();
	
	for (; it != newStateGradient->end(); it ++)
	{
		CFeatureList::iterator it2 = eTraceList->begin();
		for (;it2 != eTraceList->end(); it2 ++)
		{
			int featx = (*it)->featureIndex;
			int featy = (*it2)->featureIndex;

			A->element(featy, featx) = A->element(featy, featx) + (*it)->factor * (*it2)->factor;
		}
	}
//	for (; it != stateDiffList->end(); it ++)
//	{
//		CFeatureList::iterator it2 = eTraceList->begin();
//		for (int i = 0; i < oldState->getNumDiscreteStates(); i ++)
//		{
//			int featx = (*it)->featureIndex;
//			int featy = oldState->getDiscreteState(i);
//
//			A->element(featy, featx) = A->element(featy, featx) + (*it)->factor * oldState->getContinuousState(i);
//		}
//	}

	CFeatureList::iterator it2 = eTraceList->begin();
	for (;it2 != eTraceList->end(); it2 ++)
	{
		int featy = (*it2)->featureIndex;
		b->element(featy) = b->element(featy) + (*it2)->factor * reward;
	}

/*	for (int i = 0; i < oldState->getNumDiscreteStates(); i ++)
	{
		int featy = oldState->getDiscreteState(i);
		b->element(featy) = b->element(featy) + oldState->getContinuousState(i) * reward;
	}*/
	
}

void CLSTDLambda::newEpisode()
{
	if (nEpisode > 0 && nUpdateEpisode > 0 &&  nEpisode % nUpdateEpisode == 0)
	{
		double error = doOptimization();
	
		printf("Error in LSTD optimization: %f\n", error);
	}
	nEpisode ++;
	
	resetETraces();
}

	
void CLSTDLambda::resetData()
{
	*A = 0;
	*b = 0;
	
	nEpisode = 1;
	
}

void CLSTDLambda::loadData(FILE *stream)
{
	fprintf(stream, "LSTD Data\n");
	fprintf(stream, "A - Matrix\n");
	
	for (int i = 0; i < A->nrows(); i ++)
	{
		for (int j = 0; j < A->ncols(); j ++)
		{
			fprintf(stream, "%f ", A->element(i,j));
		}
		fprintf(stream, "\n");
	}
	
	fprintf(stream, "b - Vector\n");
	for (int j = 0; j < b->nrows(); j ++)
	{
		fprintf(stream, "%f ", b->element(j));
	}
	fprintf(stream, "\n");
}

void CLSTDLambda::saveData(FILE *stream)
{
	fscanf(stream, "LSTD Data\n");
	fscanf(stream, "A - Matrix\n");
	
	double dBuf;
	for (int i = 0; i < A->nrows(); i ++)
	{
		for (int j = 0; j < A->ncols(); j ++)
		{
			fscanf(stream, "%lf ", &dBuf);
			A->element(i,j) = dBuf;
		}
		fscanf(stream, "\n");
	}
	
	fscanf(stream, "b - Vector\n");
	for (int j = 0; j < b->nrows(); j ++)
	{
		fscanf(stream, "%lf ", &dBuf);
		b->element(j) = dBuf;
	}
	fscanf(stream, "\n");
}

CVLSTDLambda::CVLSTDLambda(CRewardFunction *rewardFunction, CFeatureVFunction *updateFunction, int nUpdatePerEpisode) : CLSTDLambda(rewardFunction, updateFunction, nUpdatePerEpisode)
{
	vFunction = updateFunction;
	
	vETraces = new CFeatureVETraces(vFunction);
	
	addParameters(vETraces);
	
}

CVLSTDLambda::~CVLSTDLambda()
{
	delete vETraces;
}


void CVLSTDLambda::getOldGradient(CStateCollection *stateCol, CAction *, CFeatureList *gradient)
{
	vFunction->getGradient(stateCol, gradient);
}

void CVLSTDLambda::getNewGradient(CStateCollection *stateCol, CFeatureList *gradient) 
{
	vFunction->getGradient(stateCol, gradient);
}
	
void CVLSTDLambda::updateETraces(CStateCollection *stateCol, CAction *action) 
{
	vETraces->updateETraces(action->getDuration());
	vETraces->addETrace(stateCol);
}

CFeatureList *CVLSTDLambda::getGradientETraces()
{
	return vETraces->getGradientETraces();
}

void CVLSTDLambda::resetETraces()
{
	vETraces->resetETraces();
}

CQLSTDLambda::CQLSTDLambda(CRewardFunction *rewardFunction, CFeatureQFunction *updateFunction, CAgentController *l_policy, int nUpdatePerEpisode) : CLSTDLambda(rewardFunction, updateFunction, nUpdatePerEpisode)
{
	qFunction = updateFunction;
	policy = l_policy;
	
	actionDataSet = new CActionDataSet(policy->getActions());
	
	qETraces = new CGradientQETraces(qFunction);
	
	addParameters(qETraces);
	
}

CQLSTDLambda::~CQLSTDLambda()
{
	delete qETraces;
	delete actionDataSet;
}


void CQLSTDLambda::getOldGradient(CStateCollection *stateCol, CAction *action, CFeatureList *gradient)
{
	qFunction->getGradient(stateCol, action, action->getActionData(), gradient);
}

void CQLSTDLambda::getNewGradient(CStateCollection *stateCol, CFeatureList *gradient) 
{
	CAction *action = policy->getNextAction(stateCol, actionDataSet);
	
	qFunction->getGradient(stateCol, action, actionDataSet->getActionData(action), gradient);
}
	
void CQLSTDLambda::updateETraces(CStateCollection *stateCol, CAction *action) 
{
	qETraces->updateETraces(action);
	qETraces->addETrace(stateCol, action);
}

CFeatureList *CQLSTDLambda::getGradientETraces()
{
	return qETraces->getGradientETraces();
}

void CQLSTDLambda::resetETraces()
{
	qETraces->resetETraces();
}
