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

#include "cpegasus.h"
#include <math.h>
#include "ctransitionfunction.h"
#include "crewardfunction.h"
#include "cevaluator.h"
#include "ccontinuousactions.h"
#include "ccontinuousactiongradientpolicy.h"
#include "ril_debug.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "cfeaturefunction.h"
#include "cagent.h"

CTransitionFunctionInputDerivationCalculator::CTransitionFunctionInputDerivationCalculator(CContinuousTimeAndActionTransitionFunction *dynModel)
{
	this->dynModel = dynModel;
}

CTransitionFunctionInputDerivationCalculator::~CTransitionFunctionInputDerivationCalculator()
{
}

CTransitionFunctionNumericalInputDerivationCalculator::CTransitionFunctionNumericalInputDerivationCalculator(CContinuousTimeAndActionTransitionFunction *dynModel, double stepSize = 0.001) : CTransitionFunctionInputDerivationCalculator(dynModel)
{
	addParameter("InputDerivationCalculatorStepSize", stepSize);

	nextState1 = new CState(dynModel->getStateProperties());
	nextState2 = new CState(dynModel->getStateProperties());

	buffState = new CState(dynModel->getStateProperties());

	buffData = dynamic_cast<CContinuousActionData *>(dynModel->getContinuousAction()->getNewActionData());

}

CTransitionFunctionNumericalInputDerivationCalculator::~CTransitionFunctionNumericalInputDerivationCalculator()
{
	delete buffState;
	delete nextState1;
	delete nextState2;

	delete buffData;

}

void CTransitionFunctionNumericalInputDerivationCalculator::getInputDerivation(CState *currentState, CContinuousActionData *data, Matrix *dModelInput)
{
	int column = 0;
	CStateProperties *stateProp = dynModel->getStateProperties();
	double stepSize = getParameter("InputDerivationCalculatorStepSize");
	for (unsigned int i = 0; i < currentState->getNumContinuousStates(); i++, column ++)
	{
		double stepSize_i = stepSize * (stateProp->getMaxValue(i) - stateProp->getMinValue(i));
		buffState->setState(currentState);

		buffState->setContinuousState(i, currentState->getContinuousState(i) - stepSize_i);
		dynModel->transitionFunction(buffState, dynModel->getContinuousAction(), nextState1, data);

		buffState->setContinuousState(i, currentState->getContinuousState(i) + stepSize_i);
		dynModel->transitionFunction(buffState, dynModel->getContinuousAction(), nextState2, data);

		for (unsigned int row = 0; row < stateProp->getNumContinuousStates(); row ++)
		{
			dModelInput->element(row, column) = (nextState2->getSingleStateDifference(row, nextState1->getContinuousState(row))) / ( 2 * stepSize_i);
		}
	}

	CContinuousActionProperties *actionProp = dynModel->getContinuousAction()->getContinuousActionProperties();

	for (int i = 0; i < data->nrows(); i++, column ++)
	{
		double stepSize_i = stepSize * (actionProp->getMaxActionValue(i) - actionProp->getMinActionValue(i));
		
		buffData->setData(data);
		buffData->element(i) = data->element(i) - stepSize_i;
		dynModel->transitionFunction(buffState, dynModel->getContinuousAction(), nextState1, buffData);

		buffData->element(i) = data->element(i) + stepSize_i;
		dynModel->transitionFunction(buffState, dynModel->getContinuousAction(), nextState2, buffData);

		for (unsigned int row = 0; row < stateProp->getNumContinuousStates(); row ++)
		{
			dModelInput->element(row, column) = (nextState2->getSingleStateDifference(row, nextState1->getContinuousState(row))) / ( 2 * stepSize_i);
		}
	}
}

CPEGASUSPolicyGradientCalculator::CPEGASUSPolicyGradientCalculator(CAgent *agent, CRewardFunction *reward, CContinuousActionGradientPolicy *policy, CTransitionFunctionEnvironment *dynModel, int numStartStates, int horizon, double gamma) 
: CPolicyGradientCalculator(policy, NULL)
{
	addParameter("DiscountFactor", gamma);
	addParameter("PEGASUSHorizon", horizon);
	addParameter("PEGASUSNumStartStates", numStartStates);
	addParameter("PEGASUSUseNewStartStates", 0.0);

	this->dynModel = dynModel;

	startStates = new CStateList(dynModel->getStateProperties());

	this->policy = policy;

	setRandomStartStates();
	
	sameStateEvaluator = new CValueSameStateCalculator(agent, reward, dynModel, startStates, (int)getParameter("PEGASUSHorizon"),getParameter("DiscountFactor"));
	evaluator = sameStateEvaluator;
	

}

CPEGASUSPolicyGradientCalculator::~CPEGASUSPolicyGradientCalculator()
{
	delete startStates;
	delete evaluator;
}

void CPEGASUSPolicyGradientCalculator::getGradient(CFeatureList *gradient)
{
	bool bUseNewStates = getParameter("PEGASUSUseNewStartStates") > 0.5;
	if (bUseNewStates || startStates->getNumStates() == 0)
	{
		setRandomStartStates();
	}
	getPEGASUSGradient(gradient, startStates);
	
	gradient->multFactor(-1.0);
}

CStateList* CPEGASUSPolicyGradientCalculator::getStartStates()
{
	return startStates;
}

void CPEGASUSPolicyGradientCalculator::setStartStates(CStateList *startStates)
{
	this->startStates->clear();
	CState *state = new CState(dynModel->getStateProperties());
	for (unsigned int i = 0; i < startStates->getNumStates(); i++)
	{
		startStates->getState(i, state);
		
		
		this->startStates->addState(state);
	}
	delete state;
}

void CPEGASUSPolicyGradientCalculator::setRandomStartStates()
{
	this->startStates->clear();
	CStateProperties *properties = dynModel->getStateProperties();
	CState *state = new CState(properties);
	for (int i = 0; i < my_round(getParameter("PEGASUSNumStartStates")); i++)
	{
		dynModel->resetModel();
		dynModel->getState(state);
		
		this->startStates->addState(state);
	}
	delete state;
}


CPEGASUSAnalyticalPolicyGradientCalculator::CPEGASUSAnalyticalPolicyGradientCalculator(CAgent *agent, CContinuousActionGradientPolicy *policy, CCAGradientPolicyInputDerivationCalculator *policydInput, CTransitionFunctionEnvironment *dynModel, CTransitionFunctionInputDerivationCalculator *dynModeldInput, CStateReward *rewardFunction, int numStartStates,int horizon, double gamma ) : CPEGASUSPolicyGradientCalculator(agent, rewardFunction, policy,dynModel, numStartStates, horizon, gamma)
{
	addParameters(policydInput, "DPolicy");
	addParameters(dynModeldInput, "DModel");

	int numActionValues = policy->getContinuousActionProperties()->getNumActionValues();

	this->rewardFunction = rewardFunction;
	this->agent = agent;
	this->dynModeldInput = dynModeldInput;
	this->policydInput = policydInput;

	this->dReward = new ColumnVector(dynModel->getNumContinuousStates());
	stateGradient1 = new std::list<CFeatureList *>();

	for (unsigned int i = 0; i < dynModel->getNumContinuousStates(); i ++)
	{
		stateGradient1->push_back(new CFeatureList());
	}

	stateGradient2 = new std::list<CFeatureList *>();

	for (unsigned int i = 0; i < dynModel->getNumContinuousStates(); i ++)
	{
		stateGradient2->push_back(new CFeatureList());
	}

	dModelGradient = new std::list<CFeatureList *>();

	for (int i = 0; i < numActionValues; i ++)
	{
		dModelGradient->push_back(new CFeatureList());
	}

	episodeGradient = new CFeatureList();

	dPolicy = new Matrix(numActionValues, dynModel->getNumContinuousStates());
	dModelInput = new Matrix(dynModel->getNumContinuousStates(), numActionValues + dynModel->getNumContinuousStates());

	steps = 0;
}

CPEGASUSAnalyticalPolicyGradientCalculator::~CPEGASUSAnalyticalPolicyGradientCalculator()
{
	delete dReward;

	std::list<CFeatureList *>::iterator it = stateGradient1->begin();

	for (; it != stateGradient1->end(); it ++)
	{
		delete *it;
	}

	delete stateGradient1;

	it = stateGradient2->begin();

	for (; it != stateGradient2->end(); it ++)
	{
		delete *it;
	}

	it = dModelGradient->begin();

	for (; it != dModelGradient->end(); it ++)
	{
		delete *it;
	}

	delete dModelGradient;

	delete episodeGradient;
	delete dPolicy;
	delete dModelInput;
}

void CPEGASUSAnalyticalPolicyGradientCalculator::getPEGASUSGradient(CFeatureList *gradientFeatures, CStateList *startStates)
{
	printf("Pegasus Gradient Evaluation\n");
	agent->addSemiMDPListener(this);
	int horizon = my_round(getParameter("PEGASUSHorizon"));
	CState *startState = new CState(dynModel->getStateProperties());
	for (unsigned int i = 0; i < startStates->getNumStates(); i ++)
	{
		printf("Evaluate Episode %d\n", i);
		agent->startNewEpisode();
		startStates->getState(i, startState);
		dynModel->setState(startState);
		
		agent->doControllerEpisode(1, horizon);

		gradientFeatures->add(episodeGradient, 1.0);
	}

	gradientFeatures->multFactor(1.0 / startStates->getNumStates());
	double norm = sqrt(gradientFeatures->multFeatureList(gradientFeatures));

	if (DebugIsEnabled())
	{
		DebugPrint('p', "Calculated Pegasus Gradient Norm: %f\n", norm);
		DebugPrint('p', "Calculated Gradient:\n");

		gradientFeatures->saveASCII(DebugGetFileHandle('p'));
	}
	printf("Finished Gradient Calculation, Gradient Norm: %f\n", norm);

	delete startState;
	agent->removeSemiMDPListener(this);
}

void CPEGASUSAnalyticalPolicyGradientCalculator::multMatrixFeatureList(Matrix *matrix, CFeatureList *features, int index, std::list<CFeatureList *> *newFeatures)
{
	CFeatureList::iterator itFeat = features->begin();

	for (; itFeat != features->end(); itFeat ++)
	{
		std::list<CFeatureList *>::iterator itList = newFeatures->begin();
		for (int row = 0; itList != newFeatures->end(); itList ++,row ++)
		{
			(*itList)->update((*itFeat)->featureIndex, (*itFeat)->factor * matrix->element(row, index));
		}
	}
}

void CPEGASUSAnalyticalPolicyGradientCalculator::nextStep(CStateCollection *oldStateCol, CAction *action, CStateCollection *newStateCol)
{
	CState *oldState = oldStateCol->getState(dynModel->getStateProperties());
	CState *nextState = newStateCol->getState(dynModel->getStateProperties());

	CContinuousActionData *data = dynamic_cast<CContinuousActionData *>(action->getActionData());

	// Clear 2nd StateGradient list
	std::list<CFeatureList *>::iterator it = stateGradient2->begin();

	for (; it != stateGradient2->end(); it ++)
	{
		(*it)->clear();
	}

	//Clear Model Gradient
	it = dModelGradient->begin();

	for (; it != dModelGradient->end(); it ++)
	{
		(*it)->clear();
	}

	// Derivation of the Reward Function
	rewardFunction->getInputDerivation(nextState, dReward);


	// Derivation of the Model
	dynModeldInput->getInputDerivation(oldState, data, dModelInput);

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "Pegasus Gradient Calculation:\n ");
		DebugPrint('p', "State Gradient:\n ");
		for (it = stateGradient1->begin(); it != stateGradient1->end(); it ++)
		{
			(*it)->saveASCII(DebugGetFileHandle('p'));
			DebugPrint('p', "\n");
		}

		DebugPrint('p', "dReward: ");
		//dReward->saveASCII(DebugGetFileHandle('p'));

		DebugPrint('p', "\n");
		DebugPrint('p',"dModel: ");
		//dModelInput->saveASCII(DebugGetFileHandle('p'));
	}

	it = stateGradient1->begin();
	for (unsigned int i = 0; i < dynModel->getNumContinuousStates(); i ++, it ++)
	{
		multMatrixFeatureList(dModelInput, *it, i, stateGradient2);
	}

	// Derivation of the policy
	policydInput->getInputDerivation(oldStateCol, dPolicy);
	
	if (DebugIsEnabled('p'))
	{
		DebugPrint('p',"dPolicy: ");
		//dPolicy->saveASCII(DebugGetFileHandle('p'));
		DebugPrint('p', "\n");
	}

	it = dModelGradient->begin();

	for (int i = 0; it != dModelGradient->end(); it++, i++)
	{
		policy->getGradient(oldStateCol, i, *it);
	}

	it = stateGradient1->begin();

	for (int i = 0; it != stateGradient1->end(); i ++, it ++)
	{
		multMatrixFeatureList(dPolicy, *it, i, dModelGradient);
	}

	it = dModelGradient->begin();

	for (int i = 0; it != dModelGradient->end(); it++, i++)
	{
		multMatrixFeatureList(dModelInput, *it, i + dynModel->getNumContinuousStates(), stateGradient2);
	}

	if (DebugIsEnabled('p'))
	{
		DebugPrint('p', "Model Gradients:\n ");
		for (it = dModelGradient->begin(); it != dModelGradient->end(); it ++)
		{
			(*it)->saveASCII(DebugGetFileHandle('p'));
			DebugPrint('p', "\n");

		}
		DebugPrint('p', "New State Gradient:\n ");

		for (it = stateGradient2->begin(); it != stateGradient2->end(); it ++)
		{
			(*it)->saveASCII(DebugGetFileHandle('p'));
			DebugPrint('p', "\n");

		}
	
	}

	
	double discountFactor = pow(getParameter("DiscountFactor"), steps);
	
	
	*dReward *= discountFactor;

	it = stateGradient2->begin();
	for (int i = 0; it != stateGradient2->end(); i ++, it ++)
	{
		episodeGradient->add(*it, dReward->element(i));
	}

	std::list<CFeatureList *> *tempGradient = stateGradient1;
	stateGradient1 = stateGradient2;
	stateGradient2 = tempGradient;

	steps ++;
}

void CPEGASUSAnalyticalPolicyGradientCalculator::newEpisode()
{
	std::list<CFeatureList *>::iterator it = stateGradient1->begin();

	for (; it != stateGradient1->end(); it ++)
	{
		(*it)->clear();
	}
	episodeGradient->clear();
	steps = 0;
}


CPEGASUSNumericPolicyGradientCalculator::CPEGASUSNumericPolicyGradientCalculator(CAgent *agent, CContinuousActionGradientPolicy *policy, CTransitionFunctionEnvironment *dynModel, CRewardFunction *rewardFunction, double stepSize, int startStates, int horizon, double gamma) : CPEGASUSPolicyGradientCalculator(agent, rewardFunction, policy, dynModel, startStates, horizon, gamma)
{
	weights = new double[policy->getNumWeights()];

	this->rewardFunction = rewardFunction;
	this->agent = agent;

	addParameter("PEGASUSNumericStepSize", stepSize);
	addParameter("DiscountFactor", gamma);
}

CPEGASUSNumericPolicyGradientCalculator::~CPEGASUSNumericPolicyGradientCalculator()
{
	delete [] weights;
}

void CPEGASUSNumericPolicyGradientCalculator::getPEGASUSGradient(CFeatureList *gradientFeatures, CStateList *startStates)
{
	sameStateEvaluator->setStartStates(startStates);

	policy->getWeights(weights);

	agent->setController(policy);

	double stepSize = getParameter("PEGASUSNumericStepSize");

	double value = evaluator->evaluatePolicy();

	for (int i = 0; i < policy->getNumWeights(); i ++)
	{
		weights[i] -= stepSize;
		policy->setWeights(weights);
		double vMinus = evaluator->evaluatePolicy();
		weights[i] += 2 * stepSize;
		policy->setWeights(weights);
		double vPlus = evaluator->evaluatePolicy();

		weights[i] -= stepSize;
		
		if (vMinus > value || vPlus > value)
		{
			gradientFeatures->set(i, (vPlus - vMinus) / (2 * stepSize));
		}
		else
		{
			gradientFeatures->set(i, 0);
		}

		printf("%f %f %f %d %d\n", stepSize, vPlus, vMinus, startStates->getNumStates(), sameStateEvaluator->getStartStates()->getNumStates());
		printf("Calculated derivation for weight %d : %f\n", i, gradientFeatures->getFeatureFactor(i));
	}
	policy->setWeights(weights);

}

