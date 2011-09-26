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

#include "chierarchicbehaviours.h"
#include "ctransitionfunction.h"
#include "cregions.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "cstateproperties.h"

#include "cutility.h"

#include "ril_debug.h"


CSubGoalBehaviour::CSubGoalBehaviour(CStateProperties *l_modelProperties, CRegion *l_avialableRegion, char *l_subgoalName) : CHierarchicalSemiMarkovDecisionProcess(l_modelProperties), CStateReward(l_modelProperties), subgoalName(l_subgoalName)
{
	standardReward = -1.0;

	availableRegion = l_avialableRegion;

	rewardFactors = new std::map<CRegion *, std::pair<double, double> >;

	targetRegions = new std::list<CRegion *>;
	failRegions = new std::list<CRegion *>;

	this->modelProperties = l_modelProperties;
}

CSubGoalBehaviour::~CSubGoalBehaviour()
{
	delete rewardFactors;
	delete targetRegions;
	delete failRegions;
}

void CSubGoalBehaviour::sendNextStep(CAction *action)
{
	bool debugPrint = false;
	if (currentSteps == 0)
	{
		debugPrint = true;
	}
	CHierarchicalSemiMarkovDecisionProcess::sendNextStep(action);

	if (debugPrint && DebugIsEnabled('h'))
	{
		DebugPrint('h', "Started SubGoal %s, ", this->subgoalName.c_str());
		currentState->getState()->saveASCII(DebugGetFileHandle('h'));
		DebugPrint('h', "\n");
	}
}


bool CSubGoalBehaviour::isFinished(CStateCollection *, CStateCollection *newState)
{
	bool finished = isInFailRegion(newState->getState(modelProperties)) || isInGoalRegion(newState->getState(modelProperties));

	if (finished)
	{
		DebugPrint('h', "Subgoal %s Finished after %d steps\n", subgoalName.c_str(), currentSteps);
	}
	return finished;
}


bool CSubGoalBehaviour::isInGoalRegion(CState *state)
{
	std::list<CRegion *>::iterator it = targetRegions->begin();

	bool finished = false;

	for (; it != targetRegions->end(); it ++)
	{
		if ((*it)->isStateInRegion(state))
		{
			finished = true;
			break;
		}
	}
	return finished;
}

bool CSubGoalBehaviour::isInFailRegion(CState *state)
{
	std::list<CRegion *>::iterator it = failRegions->begin();

	bool finished = false;

	for (; it != failRegions->end(); it ++)
	{
		if ((*it)->isStateInRegion(state))
		{
			finished = true;
			break;
		}
	}
	return finished;
}

bool CSubGoalBehaviour::isAvailable(CStateCollection *currentState)
{
	bool availAble = availableRegion->isStateInRegion(currentState->getState(modelProperties));
	availAble = availAble && ! isFinished(NULL, currentState); 

	if (availAble)
	{
		DebugPrint('h', "Subgoal %s is availAble\n", subgoalName.c_str());
	}
	else
	{
		DebugPrint('h', "Subgoal %s is not availAble\n", subgoalName.c_str());
	}

	return availAble;
}

	
double CSubGoalBehaviour::getStateReward(CState *modelState)
{
	double reward = standardReward;

	std::list<CRegion *>::iterator it = targetRegions->begin();
	
	for (; it != targetRegions->end(); it ++)
	{
		double rewardFactor = (*rewardFactors)[*it].first;
		double rewardTau = (*rewardFactors)[*it].second;

		double distance = (*it)->getDistance(modelState);
		double dReward = rewardFactor * my_exp(- rewardTau * distance);
		reward += dReward;
	}

	it = failRegions->begin();

	for (; it != failRegions->end(); it ++)
	{
		double rewardFactor = (*rewardFactors)[*it].first;
		double rewardTau = (*rewardFactors)[*it].second;

		double dReward = rewardFactor * my_exp(- rewardTau * (*it)->getDistance(modelState));

		reward += dReward;
	}
	return reward;
}

void CSubGoalBehaviour::getInputDerivation(CState *, ColumnVector *)
{
	
}

void CSubGoalBehaviour::addTargetRegion(CRegion *target, double rewardFactor , double rewardTau )
{
	targetRegions->push_back(target);

	(*rewardFactors)[target] = std::pair<double,double>(rewardFactor, rewardTau);
}

void CSubGoalBehaviour::addFailRegion(CRegion *target, double rewardFactor, double rewardTau)
{
	failRegions->push_back(target);

	(*rewardFactors)[target] = std::pair<double,double>(rewardFactor, rewardTau);
}

void CSubGoalBehaviour::setRewardFactor(CRegion *region, double rewardFactor)
{
	(*rewardFactors)[region].first = rewardFactor;
}

void CSubGoalBehaviour::setRewardTau(CRegion *region, double rewardTau)
{
	(*rewardFactors)[region].second = rewardTau;
}

/*
CSubGoalTrainer::CSubGoalTrainer(CTransitionFunction *transitionFunction, CSubGoalBehaviour *l_subGoal) : CTransitionFunctionEnvironment(transitionFunction)
{
	this->subGoal = l_subGoal;
	if (subGoal)
	{
		this->sampleRegion = l_subGoal->getAvailAbleRegion();
	}
}

void CSubGoalTrainer::doNextState(CPrimitiveAction *action)
{
	CTransitionFunctionEnvironment::doNextState(action);

	if (subGoal)
	{
		failed = failed | subGoal->isInFailRegion(this->modelState);
		reset = reset | failed | subGoal->isInGoalRegion(this->modelState);
	}
}

void CSubGoalTrainer::doResetModel()
{
	if (subGoal)
	{
		sampleRegion->getRandomStateSample(modelState);
	}
	else
	{
		CTransitionFunctionEnvironment::doResetModel();
	}
}

void CSubGoalTrainer::setSubGoal(CSubGoalBehaviour *subGoal)
{
	this->subGoal = subGoal;
	this->sampleRegion =  subGoal->getAvailAbleRegion();
}

void CSubGoalTrainer::setSampleRegion(CRegion *l_sampleRegion)
{
	sampleRegion = l_sampleRegion;
}
*/ 

CSubGoalOutput::CSubGoalOutput(CAgentController *policy)
{
	lastAction = NULL;
	this->policy = policy;
}

void CSubGoalOutput::nextStep(CStateCollection *, CAction *action, CStateCollection *newState)
{
	lastAction = (CSubGoalBehaviour *) action;
	printf("Subgoal %s ended after %d steps in State\n", lastAction->getSubGoalName().c_str(), lastAction->getDuration());
	newState->getState()->saveASCII(stdout);
	printf("Subgoal %s begins\n", ((CSubGoalBehaviour *)policy->getNextAction(newState))->getSubGoalName().c_str());

}

void CSubGoalOutput::newEpisode()
{
}


CSubGoalController::CSubGoalController(CActionSet *hierarchicActions) : CAgentController(hierarchicActions)
{

}

CAction *CSubGoalController::getNextAction(CStateCollection *state, CActionDataSet *)
{
	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it++)
	{
		if ((*it)->isAvailable(state))
		{
			return *it;
		}
	}
	printf("SubGoal Controller, fatal error: No Subgoal available!!, State: ");
	state->getState()->saveASCII(stdout);
	printf("\n");
	assert(false);
	return NULL;
}

CExtendedPrimitiveAction::CExtendedPrimitiveAction(CAction *l_primitiveAction, int l_extendedActionDuration)
{
	this->primitiveAction = l_primitiveAction;
	this->extendedActionDuration = l_extendedActionDuration;
	this->sendIntermediateSteps = false;
}

bool CExtendedPrimitiveAction::isFinished(CStateCollection *, CStateCollection *)
{
	return getDuration() >= extendedActionDuration;
}

CAction* CExtendedPrimitiveAction::getNextHierarchyLevel(CStateCollection *, CActionDataSet *)
{
	return primitiveAction;
}

CPrimitiveActionStateChange::CPrimitiveActionStateChange(CPrimitiveAction *action, CStateProperties *stateToChange)
{
	this->primitiveAction = action;
	this->stateToChange = stateToChange;
}

CAction* CPrimitiveActionStateChange::getNextHierarchyLevel(CStateCollection *, CActionDataSet *)
{
	return primitiveAction;
}

bool CPrimitiveActionStateChange::isFinished(CStateCollection *oldStateCol, CStateCollection *newStateCol)
{
	CState *oldState = oldStateCol->getState(stateToChange);
	CState *newState = newStateCol->getState(stateToChange);
	return oldState->equals(newState);
}

void CPrimitiveActionStateChange::setStateToChange(CStateProperties *stateToChange)
{
	this->stateToChange = stateToChange;
}
