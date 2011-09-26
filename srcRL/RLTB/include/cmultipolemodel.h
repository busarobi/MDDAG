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

#ifndef CMULTIPOLEMODEL_H
#define CMULTIPOLEMODEL_H

#include "cenvironmentmodel.h"
#include "crewardfunction.h"
#include "cagentcontroller.h"
#include "cdiscretizer.h"
#include "caction.h"

#include <math.h>

class CMultiPoleDiscreteState : public CAbstractStateDiscretizer
{
public:
		CMultiPoleDiscreteState();
		virtual ~CMultiPoleDiscreteState() {};

		virtual unsigned int getDiscreteStateNumber(CStateCollection *state);		
};

class CMultiPoleFailedState : public CAbstractStateDiscretizer
{
public: 
	CMultiPoleFailedState();
	virtual ~CMultiPoleFailedState() {};

	virtual unsigned int getDiscreteStateNumber(CStateCollection *state);		
};

class CMultiPoleModel : public CEnvironmentModel, public CRewardFunction
{
protected:
	/// internal state variables
	double x, x_dot, theta, theta_dot; 
	/// calculate the next state based on the action
	virtual void doNextState(CPrimitiveAction *action); 

public:
	CMultiPoleModel();
	virtual ~CMultiPoleModel();
	
	///returns the reward for the transition, implements the CRewardFunction interface
	virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState); 
	///fetches the internal state and stores it in the state object
	virtual void getState(CState *state);
	///resets the model
	virtual void doResetModel();
};

class CMultiPoleContinuousReward : public CStateReward
{
public:
	CMultiPoleContinuousReward(CStateProperties *modelState);
	virtual ~CMultiPoleContinuousReward() {};

	double getStateReward(CState *modelState);
};


class CMultiPoleAction : public CPrimitiveAction
{
protected:
	double force;

public:
	CMultiPoleAction(double force);

	double getForce();
};


class CMultiPoleController : public CAgentController
{
protected:

public:
	CMultiPoleController(CActionSet *actions);
	virtual ~CMultiPoleController();

	virtual CAction* getNextAction(CStateCollection *state, CActionDataSet *data = NULL);
};

class CMultiPoleDiscreteController : public CAgentController, CStateObject
{
protected:

public:
	CMultiPoleDiscreteController(CActionSet *actions, CStateProperties *discState);
	virtual ~CMultiPoleDiscreteController();

	virtual CAction* getNextAction(CStateCollection *state, CActionDataSet *data = NULL);
};


#endif


