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
#include "cmultipolemodel.h"
#include "cstatecollection.h"
#include "cstate.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define GRAVITY 9.8
#define MASSCART 1.0
#define MASSPOLE 0.1
#define TOTAL_MASS (MASSPOLE + MASSCART)
#define LENGTH 0.5		  /* actually half the pole's length */
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10.0
#define TAU 0.02		  /* seconds between state updates */
#define FOURTHIRDS 1.3333333333333


#define one_degree 0.0174532	/* 2pi/360 */
#define six_degrees 0.1047192
#define twelve_degrees 0.2094384
#define fifty_degrees 0.87266

CMultiPoleModel::CMultiPoleModel() : CEnvironmentModel(4, 0)
{
	x= x_dot = theta = theta_dot = 0;

	properties->setMinValue(0, -2.4 * 1.1);
	properties->setMaxValue(0, 2.4 * 1.1);

	properties->setMinValue(1, -2);
	properties->setMaxValue(1, 2);

	properties->setMinValue(2, -twelve_degrees * 1.1);
	properties->setMaxValue(2, twelve_degrees * 1.1);

	properties->setMinValue(3, -fifty_degrees * 1.5);
	properties->setMaxValue(3, fifty_degrees * 1.5);
}

CMultiPoleModel::~CMultiPoleModel() {
}

void CMultiPoleModel::doNextState(CPrimitiveAction *act)
{
	double xacc,thetaacc,force,costheta,sintheta,temp;
	// cast the action to CMultiPoleAction
    CMultiPoleAction* action = (CMultiPoleAction*)(act);
	// determine the force    
	force = action->getForce();

	// calculate the new state
    costheta = cos(theta);
    sintheta = sin(theta);
    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS;
    thetaacc = (GRAVITY * sintheta - costheta* temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS));
    xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS;
    /*** Update the four state variables, using Euler's method. ***/
    x  += TAU * x_dot;
    x_dot += TAU * xacc;
    theta += TAU * theta_dot;
    theta_dot += TAU * thetaacc;

	// determine wether the episode has failed
    if (x < -2.4 ||
          x > 2.4  ||
          theta < -twelve_degrees ||
          theta > twelve_degrees) {
          reset = true;
          failed = true;
    }
    // indicate that a new episode has begun
	if (reset)
	{
		printf("Failed State: x = %f; theta = %f\n", x, theta);
	}
}

double CMultiPoleModel::getReward(CStateCollection *, CAction *, CStateCollection *newStateCol) {
    double rew = 0.0;
    CState *newState = newStateCol->getState(getStateProperties());
	
    // calculate the reward:
    // -1: for failed
    // 0 : else
    if (newState->isResetState())
    {
	rew = - 1.0;
    }
    else rew = 0.0;

    return rew;
}

CMultiPoleContinuousReward::CMultiPoleContinuousReward(CStateProperties *modelState) : CStateReward(modelState)
{
}

double CMultiPoleContinuousReward::getStateReward(CState *modelState)
{
	double reward = 0.0;
	double theta = modelState->getContinuousState(2);
	double x = modelState->getContinuousState(0);
	reward = -fabs(theta) * 5;
	if (fabs(x) > 2.4)
	{
		reward -= 5;
	}
	return theta;
}


// Store the model state to the given state object
void CMultiPoleModel::getState(CState *state)
{
	// initializes the state object
	CEnvironmentModel::getState(state);

	// Set the 4 internal state variables to the 
	// continuous state variables of the model state
	state->setContinuousState(0, x);
	state->setContinuousState(1, x_dot);
	state->setContinuousState(2, theta);
	state->setContinuousState(3, theta_dot);
}

void CMultiPoleModel::doResetModel()
{
    /// Reset internal state variables
	x = x_dot = theta = theta_dot = 0;
}

CMultiPoleDiscreteState::CMultiPoleDiscreteState() : CAbstractStateDiscretizer(163)
{
}

unsigned int CMultiPoleDiscreteState::getDiscreteStateNumber(CStateCollection *stateCol)
{
	// get the model state
	CState *state = stateCol->getState();
	int box;
	// get the 4 continuous state variables
	double x = state->getContinuousState(0);
    double x_dot = state->getContinuousState(1);
    double theta = state->getContinuousState(2);
    double theta_dot = state->getContinuousState(3);

    if (x < -2.4 ||  x > 2.4  || theta < -twelve_degrees || theta > twelve_degrees)
	{
		box = -1; /* to signal failure */
    }
    else
	{	
		//partition x
		if (x < -0.8) box = 0;
		else if (x < 0.8) box = 1;
		else box = 2;

		//partition x_dot
		if (x_dot < -0.5);
		else if (x_dot < 0.5) box += 3;
		else box += 6;

		//partition theta
		if (theta < -six_degrees);
		else if (theta < -one_degree) box += 9;
		else if (theta < 0) box += 18;
		else if (theta < one_degree) box += 27;
		else if (theta < six_degrees) box += 36;
		else box += 45;

		//partition theta_dot
		if (theta_dot < -fifty_degrees);
		else if (theta_dot < fifty_degrees)  box += 54;
		else box += 108;
    }
	//increase box because only positiv values are allowed.
	box ++;
    
	return box;
}

CMultiPoleFailedState::CMultiPoleFailedState() : CAbstractStateDiscretizer(2)
{
}

unsigned int CMultiPoleFailedState::getDiscreteStateNumber(CStateCollection *stateCol)
{
	// get the model state
	CState *state = stateCol->getState();
	int box;
	double x = state->getContinuousState(0);
    double theta = state->getContinuousState(2);

	/// calculate wether the state is a failed state
	if (x < -2.4 ||  x > 2.4  || theta < -twelve_degrees || theta > twelve_degrees)
	{
		box = 0; /* to signal failure */
    }
	else
	{
		box = 1;
	}
	return box;
}

CMultiPoleAction::CMultiPoleAction(double force) : CPrimitiveAction()
{
    this->force = force;
}

double CMultiPoleAction::getForce()
{
    return this->force;
}


CMultiPoleController::CMultiPoleController(CActionSet *actions) : CAgentController(actions)
{
}

CMultiPoleController::~CMultiPoleController()
{
}

CAction* CMultiPoleController::getNextAction(CStateCollection *stateCol, CActionDataSet *)
{
	CState *state = stateCol->getState();

	//double x = state->getContinuousState(0);
    //double x_dot = state->getContinuousState(1);
    double theta = state->getContinuousState(2);
    double theta_dot = state->getContinuousState(3);
    double costheta = cos(theta);
    double sintheta = sin(theta);

	theta += TAU * theta_dot;
	double temp1 = (FORCE_MAG + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS;
    double theta_acc1 = (GRAVITY * sintheta - costheta* temp1) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS));
	//double x_acc1  = temp1 - POLEMASS_LENGTH * theta_acc1 * costheta / TOTAL_MASS;
	double theta1 = theta + TAU * theta_dot + TAU * (theta_dot + TAU * theta_acc1);
    //double x1 = x + TAU * x_dot + TAU * (x_dot + TAU * x_acc1);

	double temp2 = (-FORCE_MAG + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS;
    double theta_acc2 = (GRAVITY * sintheta - costheta* temp2) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS));
	//double x_acc2  = temp2 - POLEMASS_LENGTH * theta_acc2 * costheta / TOTAL_MASS;
	double theta2 = theta + TAU * theta_dot + TAU * (theta_dot + TAU * theta_acc2);
    //double x2 = x + TAU * x_dot + TAU * (x_dot + TAU * x_acc2);

	int index;
	if (fabs(theta1) >= fabs(theta2))
	{
		index = 1;
	}
	else
	{
		index = 0;
	}
	
	return this->actions->get(index);
}

CMultiPoleDiscreteController::CMultiPoleDiscreteController(CActionSet *actions, CStateProperties *discState) : CAgentController(actions), CStateObject(discState) 
{
}

CMultiPoleDiscreteController::~CMultiPoleDiscreteController()
{
}

CAction* CMultiPoleDiscreteController::getNextAction(CStateCollection *stateCol, CActionDataSet *)
{
	CState *state = stateCol->getState(properties);

	int actionIndeces[163];
	memset(actionIndeces, 0, sizeof(int) * 163);
	
	actionIndeces[2] = 1;
	actionIndeces[9] = 1;
	actionIndeces[21] = 1;
	actionIndeces[31] = 1;
	actionIndeces[32] = 1;
	actionIndeces[34] = 1;
	actionIndeces[51] = 1;
	actionIndeces[52] = 1;
	actionIndeces[53] = 1;
	actionIndeces[68] = 1;
	actionIndeces[71] = 1;
	actionIndeces[73] = 1;
	actionIndeces[75] = 1;
	actionIndeces[76] = 1;
	actionIndeces[77] = 1;
	actionIndeces[78] = 1;
	actionIndeces[79] = 1;
	actionIndeces[81] = 1;
	actionIndeces[85] = 1;
	actionIndeces[86] = 1;
	actionIndeces[87] = 1;
	actionIndeces[88] = 1;
	actionIndeces[89] = 1;
	actionIndeces[91] = 1;
	actionIndeces[93] = 1;
	actionIndeces[94] = 1;
	actionIndeces[95] = 1;
	actionIndeces[96] = 1;
	actionIndeces[97] = 1;
	actionIndeces[98] = 1;
	actionIndeces[99] = 1;
	actionIndeces[100] = 1;
	actionIndeces[101] = 1;
	actionIndeces[102] = 1;
	actionIndeces[103] = 1;
	actionIndeces[104] = 1;
	actionIndeces[105] = 1;
	actionIndeces[106] = 1;
	actionIndeces[107] = 1;
	actionIndeces[109] = 1;
	actionIndeces[110] = 1;
	actionIndeces[112] = 1;

	actionIndeces[117] = 1;
	actionIndeces[118] = 1;
	actionIndeces[119] = 1;
	actionIndeces[120] = 1;
	actionIndeces[122] = 1;
	actionIndeces[125] = 1;
	actionIndeces[127] = 1;
	actionIndeces[128] = 1;
	actionIndeces[129] = 1;
	actionIndeces[130] = 1;
	actionIndeces[131] = 1;
	actionIndeces[135] = 1;
	actionIndeces[136] = 1;
	actionIndeces[137] = 1;
	actionIndeces[138] = 1;
	actionIndeces[139] = 1;
	actionIndeces[140] = 1;
	actionIndeces[144] = 1;
	actionIndeces[145] = 1;
	actionIndeces[146] = 1;
	actionIndeces[147] = 1;
	actionIndeces[148] = 1;
	actionIndeces[149] = 1;
	actionIndeces[151] = 1;
	actionIndeces[152] = 1;
	actionIndeces[153] = 1;
	actionIndeces[154] = 1;
	actionIndeces[155] = 1;
	actionIndeces[156] = 1;
	actionIndeces[157] = 1;
	actionIndeces[161] = 1;

	int discreteStateNum = state->getDiscreteState(0);
	if (discreteStateNum == 0)
	{
		return actions->get(0);
	}
	return actions->get((actionIndeces[discreteStateNum - 1] + 1)%2);
}

