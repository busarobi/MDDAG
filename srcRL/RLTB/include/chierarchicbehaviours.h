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

#ifndef C_HIERARCHICBEHAVIOUR__H
#define C_HIERARCHICBEHAVIOUR__H

#include "cagent.h"
#include "crewardfunction.h"
#include "caction.h"
#include "cagentcontroller.h"


class CRegion;
class CStateProperties;
class CStateCollection;
class CState;


class CSubGoalBehaviour : public CHierarchicalSemiMarkovDecisionProcess, public CStateReward
{
protected:
	std::map<CRegion *, std::pair<double, double> > *rewardFactors;
	
	std::list<CRegion *> *targetRegions;
	std::list<CRegion *> *failRegions;

	double standardReward;

	CRegion *availableRegion;
	CStateProperties *modelProperties;

	string subgoalName;

public:
	CSubGoalBehaviour(CStateProperties *modelProperties, CRegion *avialableRegion, char *subgoalName = "");
	virtual ~CSubGoalBehaviour();

	virtual bool isFinished(CStateCollection *oldState, CStateCollection *newState);
	virtual bool isAvailable(CStateCollection *currentState);

	virtual bool isInGoalRegion(CState *state);
	virtual bool isInFailRegion(CState *state);

	virtual double getStateReward(CState *modelState);
	virtual void getInputDerivation(CState *modelState, ColumnVector *targetState);

	virtual void addTargetRegion(CRegion *target, double rewardFactor = 1.0, double rewardTau = 10);
	virtual void addFailRegion(CRegion *target, double rewardFactor = - 1.0, double rewardTau = 10);

	void setRewardFactor(CRegion *region, double rewardFactor);
	void setRewardTau(CRegion *region, double rewardTau);

	void setStandardReward(double l_standardReward) {this->standardReward = l_standardReward;};

	virtual CRegion *getAvailAbleRegion() {return availableRegion;};

	virtual void sendNextStep(CAction *action);

	string getSubGoalName() {return subgoalName;};
};

class CSubGoalController : public CAgentController
{
protected:
public:
	CSubGoalController(CActionSet *hierarchicActions);

	virtual CAction *getNextAction(CStateCollection *state, CActionDataSet *data = NULL);
};


class CSubGoalOutput : public CSemiMDPListener
{
protected:
	CSubGoalBehaviour *lastAction;
	CAgentController *policy;

public:
	CSubGoalOutput(CAgentController *policy);

	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState);
	virtual void newEpisode();
};
/*
class CSubGoalTrainer : public CTransitionFunctionEnvironment
{
protected:
	CSubGoalBehaviour *subGoal;
	CRegion *sampleRegion;
public:
	CSubGoalTrainer(CTransitionFunction *transitionFunction, CSubGoalBehaviour *subGoal);

	virtual void doNextState(CPrimitiveAction *action);
	virtual void doResetModel();

	virtual void setSubGoal(CSubGoalBehaviour *subGoal);
	virtual void setSampleRegion(CRegion *l_sampleRegion);

};
*/

class CExtendedPrimitiveAction : public CExtendedAction
{
protected:
	CAction *primitiveAction;
public:
	int extendedActionDuration;

	CExtendedPrimitiveAction(CAction *primitiveAction, int extendedActionDuration);

	virtual bool isFinished(CStateCollection *oldState, CStateCollection *newState);

	virtual CAction* getNextHierarchyLevel(CStateCollection *state, CActionDataSet *actionDataSet = NULL);


};

/// This class represents an primitive action which gets executed until a specific (mostly discrete) state changes.
/**
This extended action subclass executes a primitive action until a specified state changes. This can be useful for example in gridworlds with local states. The state which has to change is given by "stateToChange" an will be most time a discrete state (continuous states or features normally always change). 
The isFinished method returns as long false as the 2 states (oldState and newState) are the same. 


*/
class CPrimitiveActionStateChange : public CExtendedAction
{
protected:
	/// The Properties of the state which has to change 
	CStateProperties *stateToChange;

	CPrimitiveAction *primitiveAction;

public:
	CPrimitiveActionStateChange(CPrimitiveAction *action, CStateProperties *stateToChange);

	/// Always returns primitiveAction.
	virtual CAction* getNextHierarchyLevel(CStateCollection *state, CActionDataSet *actionDataSet = NULL);

	// Returns true if the 2 states are not equal
	virtual bool isFinished(CStateCollection *oldState, CStateCollection *newState);

	// Sets the state which has to change
	void setStateToChange(CStateProperties *stateToChange);

};


#endif

