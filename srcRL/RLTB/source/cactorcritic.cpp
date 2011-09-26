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
#include "cactorcritic.h"
#include "cpolicies.h"
#include <math.h>

#include "cqfunction.h"
#include "cvfunction.h"
#include "cvfunctionlearner.h"
#include "cvetraces.h"
#include "cqetraces.h"
#include "cparameters.h"
#include "ccontinuousactiongradientpolicy.h"
#include "ril_debug.h"
#include "caction.h"


CActor::CActor( )
{
	addParameter("ActorLearningRate",0.2);
}

double CActor::getLearningRate()
{
	return getParameter("ActorLearningRate");
}

void CActor::setLearningRate(double learningRate)
{
	setParameter("ActorLearningRate", learningRate);
}

	
CActorFromQFunction::CActorFromQFunction(CAbstractQFunction *qFunction) : CActor()
{
	this->qFunction = qFunction;
	eTraces = qFunction->getStandardETraces();

	addParameters(qFunction);
	addParameters(eTraces, "Actor");

}

CActorFromQFunction::~CActorFromQFunction()
{
	delete eTraces;
}

void CActorFromQFunction::receiveError(double critic, CStateCollection *state, CAction *action, CActionData *)
{
	DebugPrint('t',"Actor updating Etraces \n");
	eTraces->updateETraces(action);
	eTraces->addETrace(state, action);
	DebugPrint('t', "Actor updating QFunction with critic %f \n", critic);
	eTraces->updateQFunction(critic * getLearningRate());
}

void CActorFromQFunction::newEpisode()
{
	eTraces->resetETraces();
}

CAbstractQFunction *CActorFromQFunction::getQFunction()
{
	return qFunction;
}

CAbstractQETraces *CActorFromQFunction::getETraces()
{
	return eTraces;
}


CActorFromQFunctionAndPolicy::CActorFromQFunctionAndPolicy(CAbstractQFunction *qFunction, CStochasticPolicy *policy) : CActorFromQFunction(qFunction)
{
	this->policy = policy;
	
	actionValues = new double[policy->getActions()->size()];

	addParameters(policy);

	addParameter("PolicyMinimumLearningRate", 0.5);
}

CActorFromQFunctionAndPolicy::~CActorFromQFunctionAndPolicy()
{
	delete actionValues;
}

void CActorFromQFunctionAndPolicy::receiveError(double critic, CStateCollection *state, CAction *Action, CActionData *)
{
	policy->getActionProbabilities(state, qFunction->getActions(), actionValues);
	double prob = actionValues[qFunction->getActions()->getIndex(Action)];

	eTraces->updateETraces(Action);
	eTraces->addETrace(state, Action, getParameter("PolicyMinimumLearningRate") + 1.0 - prob);
	eTraces->updateQFunction(critic * getLearningRate());
}

CActorFromActionValue::CActorFromActionValue(CAbstractVFunction *vFunction, CAction *action1, CAction *action2) : CAgentController(new CActionSet())
{
	actions->add(action1);
	actions->add(action2);

	addParameters(vFunction);

	this->vFunction = vFunction;
	this->eTraces = vFunction->getStandardETraces();

	addParameters(eTraces, "Actor");

	setParameter("ActorLearningRate", 1000.0);
}

CActorFromActionValue::~CActorFromActionValue()
{
	delete actions;
	delete eTraces;
}

void CActorFromActionValue::receiveError(double critic, CStateCollection *oldState, CAction *action, CActionData *)
{
	int actionIndex = actions->getIndex(action);
	eTraces->updateETraces(action->getDuration());
	eTraces->addETrace(oldState, (actionIndex == 0) - 0.5);
	eTraces->updateVFunction(critic * getLearningRate());
}

CAction *CActorFromActionValue::getNextAction(CStateCollection *state, CActionDataSet *)
{
	double value = vFunction->getValue(state);

	if (value > 50)
	{
		value = 50;
	}
	if (value < -50)
	{
		value = -50;
	}

	double propability = 1.0 / (1.0 + exp(- value));
	int actionIndex = ((double) rand() / (double) RAND_MAX) > propability;

	return actions->get(actionIndex);
}

void CActorFromActionValue::newEpisode()
{
	eTraces->resetETraces();
}

CActorFromContinuousActionGradientPolicy::CActorFromContinuousActionGradientPolicy(CContinuousActionGradientPolicy *l_gradientPolicy)
{
	this->gradientPolicy = l_gradientPolicy;
	gradientETraces = new CGradientVETraces(NULL);
	gradientFeatureList = new CFeatureList();

	addParameters(gradientPolicy);
	addParameters(gradientETraces, "Actor");

	policyDifference = new CContinuousActionData(l_gradientPolicy->getContinuousActionProperties());
}

CActorFromContinuousActionGradientPolicy::~CActorFromContinuousActionGradientPolicy()
{
	delete gradientETraces;
	delete gradientFeatureList;
	delete policyDifference;
}

void CActorFromContinuousActionGradientPolicy::receiveError(double critic, CStateCollection *oldState, CAction *Action, CActionData *data)
{
	gradientETraces->updateETraces(Action->getDuration());
	
	CContinuousActionData *contData = NULL;
	if (data)
	{
		contData = dynamic_cast<CContinuousActionData *>(data);
	}
	else
	{
		contData = dynamic_cast<CContinuousActionData *>(Action->getActionData());

	}

	assert(gradientPolicy->getRandomController());
	ColumnVector *noise = gradientPolicy->getRandomController()->getLastNoise();

	if (DebugIsEnabled('a'))
	{
		DebugPrint('a', "ActorCritic Noise: ");
		policyDifference->saveASCII(DebugGetFileHandle('a'));
	}

	for (int i = 0; i < gradientPolicy->getNumOutputs(); i ++)
	{
		gradientFeatureList->clear();
		gradientPolicy->getGradient(oldState, i, gradientFeatureList);
		
		gradientETraces->addGradientETrace(gradientFeatureList, noise->element(i));
	}

	gradientPolicy->updateGradient(gradientETraces->getGradientETraces(), critic * getParameter("ActorLearningRate"));
}

void CActorFromContinuousActionGradientPolicy::newEpisode()
{
	gradientETraces->resetETraces();
}

/*
CActorCriticLearner::CActorCriticLearner(CRewardFunction *rewardFunction, CActor *actor, CAbstractVFunction *critic) : CSemiMDPRewardListener(rewardFunction)
{
	this->actor = actor;
	this->critic = critic;

	addParameters(critic);
	addParameters(actor);

	addParameter("DiscountFactor", 0.95);
}


void CActorCriticLearner::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	double td = 0.0;

	td = getTemporalDifference(oldState, action, reward, nextState);

	actor->receiveError(td, oldState, action, nextState);
}

double CActorCriticLearner::getTemporalDifference(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	return reward + pow(getParameter("DiscountFactor"), action->getDuration()) * critic->getValue(nextState) - critic->getValue(oldState);
}

CAbstractVFunction *CActorCriticLearner::getCritic()
{
	return critic;
}

CActor *CActorCriticLearner::getActor()
{
	return actor;
}

void CActorCriticLearner::newEpisode()
{
	actor->newEpisode();

}
*/


CActorForMultipleAgents::CActorForMultipleAgents(CActionSet *actions) : CAgentController(actions)
{
	actors = new std::list<CActor *>();
	actionSets = new std::list<CAgentController *>();
	numActions  = 1;
}

CActorForMultipleAgents::~CActorForMultipleAgents()
{
	delete actors;
	delete actionSets;
}

void CActorForMultipleAgents::addActor(CActor *actor, CAgentController *policy)
{
	actors->push_back(actor);
	actionSets->push_back(policy);
	
	numActions = numActions * policy->getActions()->size();
	
	assert(numActions <= actions->size());
	
	addParameters(actor);
	addParameters(policy);
}

void CActorForMultipleAgents::receiveError(double critic, CStateCollection *state, CAction *action,  CActionData *data)
{
	int actionIndex = actions->getIndex(action);
	
	std::list<CActor *>::iterator it = actors->begin();
	std::list<CAgentController *>::iterator it2 = actionSets->begin();
	
	
	for (; it != actors->end(); it++, it2 ++)
	{
		int l_index = actionIndex % (*it2)->getActions()->size();	
		CAction *l_action = (*it2)->getActions()->get(l_index);
		(*it)->receiveError(critic, state, l_action , data);
		actionIndex = actionIndex / (*it2)->getActions()->size();	
	}
}

CAction* CActorForMultipleAgents::getNextAction(CStateCollection *state, CActionDataSet *)
{
	int actionIndex = 0;
	int actionDim = 1;
	
	std::list<CActor *>::iterator it = actors->begin();
	std::list<CAgentController *>::iterator it2 = actionSets->begin();
	
	
	for (; it != actors->end(); it++, it2 ++)
	{
		CAction *l_action = (*it2)->getNextAction(state, NULL);
		int l_index = (*it2)->getActions()->getIndex(l_action);
		
		//printf("Actor %d choosed Action %d\n", i, l_index);
		
		actionIndex += l_index * actionDim;
		
		actionDim = actionDim * (*it2)->getActions()->size();	
	}
	//printf("Choosed Action %d\n", actionIndex);
	
	return actions->get(actionIndex);
}

