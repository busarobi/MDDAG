// Copyright (C) 2003
// Gerhard Neumann (gneumann@gmx.net)
// Stephan Neumann (sneumann@gmx.net) 
//                
// This file is part of RL Toolbox.
// http://www.igi.tugraz.at/ril_toolbox
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "ril_debug.h"
#include "cpolicies.h"
#include "cutility.h"
#include "ctheoreticalmodel.h"
#include "cqfunction.h"
#include "cvfunction.h"
#include "cactionstatistics.h"
#include "cstateproperties.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "ctransitionfunction.h"
#include "cfeaturefunction.h"


#include <assert.h>
#include <math.h>

CQGreedyPolicy::CQGreedyPolicy(CActionSet *actions, CAbstractQFunction *qFunction) : CAgentController(actions)
{
	this->qFunction = qFunction;
	availableActions = new CActionSet();
	
}

CQGreedyPolicy::~CQGreedyPolicy()
{
	delete availableActions;
}


CAction *CQGreedyPolicy::getNextAction(CStateCollection *state, CActionDataSet *data )
{
	getActions()->getAvailableActions(availableActions, state);	
	return qFunction->getMax(state, availableActions, data);
}

void CActionDistribution::getGradientFactors(CStateCollection *, CAction *, CActionSet *, double *, ColumnVector *) 
{
}

CSoftMaxDistribution::CSoftMaxDistribution(double beta)
{
	addParameter("SoftMaxBeta", beta);
}

void CSoftMaxDistribution::getDistribution(CStateCollection *, CActionSet *availableActions, double *values)
{
	double sum = 0.0;
	double beta = getParameter("SoftMaxBeta");
	unsigned int i;
	unsigned int numValues = availableActions->size();

	double minValue = values[0];
	double maxValue = values[0];

	for (i = 1; i < numValues; i++)
	{
		if (minValue > values[i])
		{
			minValue = values[i]; 
		}
		if (maxValue < values[i])
		{
			maxValue = values[i];
		}
	}

	if (beta * (maxValue - minValue) > MAX_EXP)
	{
		beta = MAX_EXP / (maxValue - minValue);
	}

	for (i = 0; i < numValues; i++)
	{
		double temp = exp(beta * (values[i] - minValue));
		
		if (temp != temp)
		{
			printf("To Large value for exp function (%f) : %f %f %f\n",temp, beta * (values[i] - minValue), values[i], minValue);
		}
		
		values[i] = temp;
		
		
		
		sum += values[i];
	}
	assert(sum > 0);
	for (i = 0; i < numValues; i++)
	{
		values[i] = values[i] / sum;
		
		if (!(values[i] >= 0 && values[i] <= 1.000001))
		{
			printf("That is not a Probability: %1.10f\n", values[i]);
		}
		assert(values[i] >= 0 && values[i] <= 1.000001);
	}
}

void CSoftMaxDistribution::getGradientFactors(CStateCollection *, CAction *usedAction, CActionSet *availableActions, double *actionValues, ColumnVector *factors) 
{
	int numValues = availableActions->size();
	double normTerm = 0.0;
	double beta = getParameter("SoftMaxBeta");
	int actIndex = availableActions->getIndex(usedAction);

	double minValue = actionValues[0];
	double maxValue = actionValues[0];

	DebugPrint('p', "SoftMax Gradient Factors:\n");

	for (int i = 0; i < numValues; i++)
	{
		if (minValue > actionValues[i])
		{
			minValue = actionValues[i]; 
		}
		if (maxValue < actionValues[i])
		{
			maxValue = actionValues[i];
		}
		DebugPrint('p', "%f ", actionValues[i]);
	}
	DebugPrint('p', "\n");


	if (beta * (maxValue - minValue) > 200)
	{
		beta = 200 / (maxValue - minValue);
	}

	for (int i = 0; i <  numValues; i++)
	{
		normTerm += exp(beta * (actionValues[i] - minValue));
	}

	double buf = exp(beta * (actionValues[actIndex] - minValue));

	DebugPrint('p', "Beta:%f\n", normTerm);

	for (int i = 0; i < numValues; i ++)
	{
		factors->element(i) = - beta * buf * exp(beta * (actionValues[i] - minValue)) / pow(normTerm, (double)2.0);
	}
	factors->element(actIndex) = factors->element(actIndex) + buf * beta / normTerm;

	DebugPrint('p', "SoftMax Gradient Factors:\n");
	for (int i = 0; i < numValues; i ++)
	{
		DebugPrint('p', "%f ", factors->element(i));
	}
	DebugPrint('p', "\n");
}

CAbsoluteSoftMaxDistribution::CAbsoluteSoftMaxDistribution(double absoluteValue)
{
	addParameter("SoftMaxAbsoluteValue", absoluteValue);
}

void CAbsoluteSoftMaxDistribution::getDistribution(CStateCollection *, CActionSet *availableActions, double *values)
{
	double sum = 0.0;
	double absoluteValue = getParameter("SoftMaxAbsoluteValue");
	unsigned int i;
	unsigned int numValues = availableActions->size();
	double beta = 0.0;

	double minValue = values[0];
	double maxValue = values[0];

	for (i = 1; i < numValues; i++)
	{
		if (minValue > values[i])
		{
			minValue = values[i]; 
		}
		if (maxValue < values[i])
		{
			maxValue = values[i];
		}
	}

	if ((fabs(maxValue) <= 0.0000001 && fabs(minValue) <= 0.0000001))
	{
		beta = 100;
	}
	else
	{
		if (fabs(maxValue) < fabs(minValue) )
		{
			beta = absoluteValue / (fabs(minValue));
		}
		else
		{
			beta = absoluteValue / (fabs(maxValue));
		}
	}

	if (beta * fabs((maxValue - minValue))  > 400)
	{
		beta = 400 / fabs(maxValue - minValue);
	}
	
	for (i = 0; i < numValues; i++)
	{
		values[i] = exp(beta * (values[i] - minValue));
		sum += values[i];
	}
	assert(sum > 0);
	for (i = 0; i < numValues; i++)
	{
		values[i] = values[i] / sum;
		assert(values[i] >= 0 && values[i] <= 1);
	}
}

void CGreedyDistribution::getDistribution(CStateCollection *, CActionSet *availableActions, double *actionValues)
{
	double max = actionValues[0];
	int maxIndex = 0;
	unsigned int numValues = availableActions->size();

	actionValues[0] = 0.0;

	for (unsigned int i = 1; i < numValues; i++)
	{
		if (actionValues[i] > max)
		{
			max = actionValues[i];
			maxIndex = i;
		}
		actionValues[i] = 0.0;
	}
	actionValues[maxIndex] = 1.0;
}

CEpsilonGreedyDistribution::CEpsilonGreedyDistribution(double epsilon)
{
	addParameter("EpsilonGreedy", epsilon);
}

void CEpsilonGreedyDistribution::getDistribution(CStateCollection *, CActionSet *availableActions, double *actionValues)
{
	unsigned int numValues = availableActions->size();
	double epsilon = getParameter("EpsilonGreedy");
	double prop = epsilon / numValues;
	double max = actionValues[0];
	int maxIndex = 0;
	
	for (unsigned int i = 0; i < numValues; i++)
	{
		if (actionValues[i] > max)
		{
			max = actionValues[i];
			maxIndex = i;
		}
		actionValues[i] = prop;
	}
	actionValues[maxIndex] += 1 - epsilon;
}

CStochasticPolicy::CStochasticPolicy(CActionSet *actions, CActionDistribution *distribution) : CAgentStatisticController(actions)
{
	actionValues = new double[actions->size()];
	this->distribution = distribution;

	addParameters(distribution);

	gradientFactors = new ColumnVector(actions->size());

	actionGradientFeatures = new CFeatureList();
	
	availableActions = new CActionSet();
}

CStochasticPolicy::~CStochasticPolicy()
{
	delete [] actionValues;
	delete gradientFactors;
	delete actionGradientFeatures;
	
	delete availableActions;
}

void CStochasticPolicy::getActionProbabilities(CStateCollection *state, CActionSet *availableActions, double *actionValues, CActionDataSet *actionDataSet)
{
	getActionValues(state, availableActions, actionValues, actionDataSet);
	distribution->getDistribution(state, availableActions, actionValues);
}

CAction *CStochasticPolicy::getNextAction(CStateCollection *state, CActionDataSet *dataSet, CActionStatistics *stat)
{

	getActions()->getAvailableActions(availableActions, state);
	assert(availableActions->size() > 0);
	
	getActionProbabilities(state, availableActions, actionValues, dataSet);

	double sum = actionValues[0];
	CActionSet::iterator it = availableActions->begin();
	double z = (double) rand() / (RAND_MAX);
	unsigned int i = 0;

	while (sum <= z && i < availableActions->size() - 1)
	{
		i++; 
		it++;
		sum += actionValues[i];
	}

    if (stat != NULL)
	{
		stat->owner = this;
		getActionStatistics(state, (*it), stat);
	}

	DebugPrint('p', "ActionPropabilities: ");
	for (unsigned int j = 0; j < availableActions->size(); j++)
	{
		DebugPrint('p', "%f ", actionValues[j]);
	}
	DebugPrint('p', "\nChoosed Action: %d\n", actions->getIndex(*it));

	CAction *action = *it;
	return action;
}

void CStochasticPolicy::getActionProbabilityGradient(CStateCollection *state, CAction *action, CActionData *, CFeatureList *gradientState)
{
	gradientState->clear();

	if (isDifferentiable())
	{
		getActionValues(state,actions, this->actionValues);
		distribution->getGradientFactors(state, action, actions, actionValues, gradientFactors);

		CActionSet::iterator it = actions->begin();

		for  (int j = 0;it != actions->end(); it ++, j++)
		{
			actionGradientFeatures->clear();
			getActionGradient(state, *it, NULL, actionGradientFeatures);	
			CFeatureList::iterator itFeat = actionGradientFeatures->begin();

			for (; itFeat != actionGradientFeatures->end(); itFeat++)
			{
				gradientState->update((*itFeat)->featureIndex, (*itFeat)->factor * gradientFactors->element(j));
			}
		}
	}

	
	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "Policy Gradient Factors:\n");
		gradientState->saveASCII(DebugGetFileHandle('p'));
		DebugPrint('p', "\n");
	}
}

void CStochasticPolicy::getActionProbabilityLnGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientState)
{
	double prop = 0.0;

	getActionProbabilities(state, actions, actionValues);

	prop = actionValues[actions->getIndex(action)];

	getActionProbabilityGradient(state, action, data, gradientState);
	
	gradientState->multFactor(1 / prop);

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "Policy Gradient Ln Factors:\n");
		gradientState->saveASCII(DebugGetFileHandle('p'));
		DebugPrint('p', "\n");
	}
}

void CStochasticPolicy::getActionGradient(CStateCollection *, CAction *, CActionData *, CFeatureList *) 
{
}

CQStochasticPolicy::CQStochasticPolicy(CActionSet *actions, CActionDistribution *distribution, CAbstractQFunction *qfunction) : CStochasticPolicy(actions, distribution)
{
	this->qfunction = qfunction;

	addParameters(qfunction);
}

CQStochasticPolicy::~CQStochasticPolicy()
{
}



bool CQStochasticPolicy::isDifferentiable()
{
	return (distribution->isDifferentiable() && qfunction->isType(GRADIENTQFUNCTION));
}

void CQStochasticPolicy::getActionGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientState)
{
	gradientState->clear();

	if (isDifferentiable())
	{
		CGradientQFunction *gradQFunc = dynamic_cast<CGradientQFunction *>(qfunction);
		gradQFunc->getGradient(state, action, data, gradientState);
	}

}


void CQStochasticPolicy::getActionStatistics(CStateCollection *state, CAction *action, CActionStatistics *stat)
{
	this->qfunction->getStatistics(state, action,qfunction->getActions(), stat);
}

void CQStochasticPolicy::getActionValues(CStateCollection *state, CActionSet *availableActions, double *actionValues,  CActionDataSet *)
{
	for (unsigned int i = 0; i < availableActions->size(); actionValues[i++] = 0.0);
	qfunction->getActionValues(state, availableActions, actionValues);
}
/*
void CQStochasticPolicy::updateGradient(CFeatureList *gradient, double factor)
{
	if (qfunction->isType(GRADIENTQFUNCTION))
	{
		CGradientQFunction *gradQFunc = dynamic_cast<CGradientQFunction *>(qfunction);
		gradQFunc->updateGradient(gradient, factor);
	}
}

int CQStochasticPolicy::getNumWeights()
{
	if (qfunction->isType(GRADIENTQFUNCTION))
	{
		return dynamic_cast<CGradientQFunction *>(qfunction)->getNumWeights();
	}
	else
	{
		return 0;
	}
}*/

CVMStochasticPolicy::CVMStochasticPolicy(CActionSet *actions, CActionDistribution *distribution, CAbstractVFunction *vFunction, CTransitionFunction *model, CRewardFunction *reward, std::list<CStateModifier *> *modifiers) : CQStochasticPolicy(actions, distribution, new CQFunctionFromTransitionFunction(actions, vFunction, model, reward, modifiers))
{
	this->vFunction = vFunction;
	this->model = model;
	this->reward = reward;

	addParameters(vFunction);

//	addParameters(model);

	addParameter("DiscountFactor", 0.95);

	nextState = new CStateCollectionImpl(model->getStateProperties());
	intermediateState = new CStateCollectionImpl(model->getStateProperties());

	nextState->addStateModifiers(modifiers);
	intermediateState->addStateModifiers(modifiers);
}

CVMStochasticPolicy::~CVMStochasticPolicy()
{
	delete nextState;
	delete intermediateState;
}


/*
void CVMStochasticPolicy::getActionValues(CStateCollection *state, CActionSet *availableActions, double *actionValues,  CActionDataSet *actionDataSet)
{
	CActionSet::iterator it = availableActions->begin();
	for (int i = 0; it != availableActions->end(); it ++, i++)
	{
		CPrimitiveAction *primAction = ((CPrimitiveAction *)(*it));
		int duration = 1;
		if (primAction->isStateToChange())
		{
			CStateCollectionImpl *buf = NULL;
			nextState->getState(model->getStateProperties())->setState(state->getState(model->getStateProperties()));

			CMultiStepActionData *data = NULL;
			if (actionDataSet)
			{
				data = dynamic_cast<CMultiStepActionData *>(actionDataSet->getActionData(*it));
			}
			duration = 0;
			do
			{
				//exchange Model State

				buf = intermediateState;
				intermediateState = nextState;
				nextState = buf;

				model->transitionFunction(intermediateState->getState(model->getStateProperties()), (*it), nextState->getState(model->getStateProperties()));
				nextState->newModelState();
				duration += primAction->getSingleExecutionDuration();
			}
			// Execute the action until the state changed
			while (duration < primAction->maxStateToChangeDuration && !primAction->stateChanged(state, nextState));
		
			if (data)
			{
				data->duration = duration;
			}
		}
		else
		{
			model->transitionFunction(state->getState(model->getStateProperties()), *it, nextState->getState(model->getStateProperties()));
			nextState->newModelState();
		}
		
		if (actionDataSet && (*it)->isType(MULTISTEPACTION))
		{
			CActionData *actionData = actionDataSet->getActionData(*it);
			CMultiStepActionData *multiStepActionData  = dynamic_cast<CMultiStepActionData *>(actionData);
			duration = multiStepActionData->duration;
		}
		else
		{
			duration = (*it)->getDuration();
		}

		double rewardValue = reward->getReward(state, *it, nextState);
		double value = vFunction->getValue(nextState);
		actionValues[i] = rewardValue + pow(getParameter("DiscountFactor"), duration) * value;

		if (DebugIsEnabled('p'))
		{
			DebugPrint('p', "VM Stochastic Policy: Action %d, State: ",i);
			nextState->getState()->saveASCII(DebugGetFileHandle('p'));
			DebugPrint('p', ", functionValue: %f, reward %f\n", value, rewardValue);
		}

	}
}*/

void CVMStochasticPolicy::getActionGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientState)
{
	gradientState->clear();

	if (isDifferentiable())
	{
		model->transitionFunction(state->getState(model->getStateProperties()), action, nextState->getState(model->getStateProperties()), data);
		nextState->newModelState();

		CGradientVFunction *gradVFunc = dynamic_cast<CGradientVFunction *>(this->vFunction);

		gradVFunc->getGradient(nextState, gradientState);

		int duration = 1;

		if (data && action->isType(MULTISTEPACTION))
		{
			duration = dynamic_cast<CMultiStepActionData *>(data)->duration;
		}
		else
		{
			duration = action->getDuration();
		}

		gradientState->multFactor(pow(getParameter("DiscountFactor"), duration));
	}
}

bool CVMStochasticPolicy::isDifferentiable()
{
	return (distribution->isDifferentiable() && vFunction->isType(GRADIENTVFUNCTION));
}

/*
CExplorationDistribution::CExplorationDistribution(CActionSet *actions, CActionDistribution *distribution, CExplorationGain *explorationGain) : CActionDistribution()
{
	this->explorationGain = explorationGain;
	this->distribution = distribution;
	
	addParameter("ExplorationFactor", 0.5);
	addParameters(distribution);
	addParameters(explorationGain);

	exploration = new double[actions->size()];

}

void CExplorationDistribution::getDistribution(CStateCollection *state, CActionSet *availableActions, double *actionValues)
{

	CActionSet::iterator it = availableActions->begin();

	double norm = 0.0;
	unsigned int i = 0;

	for (i = 0, it = availableActions->begin(); i < availableActions->size(); i++, it++)
	{
		exploration[i] = explorationGain->getExplorationGain(*it, state);
		norm += exploration[i];
	}

	distribution->getDistribution(state, availableActions,actionValues);

	double alpha = getAlpha(state);

	for (i = 0; i < availableActions->size(); i++)
	{
		actionValues[i] = alpha * exploration[i] + (1 - alpha) * actionValues[i];
	}
}

void CExplorationDistribution::getGradientFactors(CStateCollection *state, CAction *usedAction, CActionSet *actions, double *actionFactors, ColumnVector *gradientFactors)
{

}

CExplorationDistribution::~CExplorationDistribution()
{
	delete exploration;
}

double CExplorationDistribution::getAlpha(CStateCollection *state)
{
	return getParameter("ExplorationFactor");
}

void CExplorationDistribution::setAlpha(double alpha)
{
	setParameter("ExplorationFactor", alpha);
}

CExplorationGain *CExplorationDistribution::getExplorationGain()
{
	return this->explorationGain;
}

void CExplorationDistribution::setExplorationGain(CExplorationGain *explorationGain)
{
	this->explorationGain = explorationGain;
}

CExplorationGain::CExplorationGain(CAbstractFeatureStochasticEstimatedModel *model, CStateModifier *calc) : CStateObject(calc)
{
	this->model = model;
}
	
double CExplorationGain::getExplorationGain(CAction *action, CStateCollection *state)
{
	CState *featState = state->getState(properties);
	int	actionIndex = model->getActions()->getIndex(action);

	double infGain = 0.0;

	int type = featState->getStateProperties()->getType() & (FEATURESTATE | DISCRETESTATE);
	switch (type)
	{
		case FEATURESTATE:
		{
			for (unsigned int i = 0; i < featState->getNumContinuousStates(); i++)
			{
				infGain += featState->getContinuousState(i) * this->getExplorationGain(actionIndex, featState->getDiscreteState(i));
			}
			break;
		}
		case DISCRETESTATE:
		{
			infGain = this->getExplorationGain(actionIndex, featState->getDiscreteState(0));
			break;
		}
		default:
		{
		}
	}

    return infGain;
}


CLogExplorationGain::CLogExplorationGain(CAbstractFeatureStochasticEstimatedModel *model, CStateModifier *calc) : CExplorationGain(model, calc)
{
}

double CLogExplorationGain::getExplorationGain(int action, int feature)
{
	return 1.0 / (log(model->getStateActionVisits(feature, action) + 1) + 1);
}

CPowExplorationGain::CPowExplorationGain(CAbstractFeatureStochasticEstimatedModel *model, CStateModifier *calc, double power) : CExplorationGain(model, calc)
{
	addParameter("PowExplorationExponent", power);
}


double CPowExplorationGain::getExplorationGain(int action, int feature)
{
	double expon =  getParameter("PowExplorationExponent");
	return  pow(model->getStateActionVisits(feature, action) + 1, expon);
}

void CPowExplorationGain::setPower(double power)
{
	setParameter("PowExplorationExponent", power);
}

double CPowExplorationGain::getPower()
{
	return getParameter("PowExplorationExponent");
}

*/
