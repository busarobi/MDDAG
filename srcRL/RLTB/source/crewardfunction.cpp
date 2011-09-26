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
#include "crewardfunction.h"
//#include "crewardmodel.h"
#include "cstatecollection.h"
#include "caction.h"
#include "cfeaturefunction.h"
#include "cvfunction.h"
#include "cstateproperties.h"
#include "cstate.h"
#include "cstatemodifier.h"


CFeatureRewardFunction::CFeatureRewardFunction(CStateProperties * discretizer) : CStateObject(discretizer)
{
	this->discretizer = discretizer;
}

CFeatureRewardFunction::~CFeatureRewardFunction()
{
	
}

double CFeatureRewardFunction::getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState)
{
	
	return getReward(oldState->getState(properties), action, newState->getState(properties));
}

double CFeatureRewardFunction::getReward(CFeatureList *oldState, CAction *action, CFeatureList *newState)
{
	double reward = 0;

	CFeatureList::iterator oldIt;
	CFeatureList::iterator newIt;

	for (oldIt = oldState->begin(); oldIt != oldState->end(); oldIt++)
	{
		for (newIt = newState->begin(); newIt != newState->end(); newIt++)
		{
			reward += getReward((*oldIt)->featureIndex, action, (*newIt)->featureIndex) * (*oldIt)->factor * (*newIt)->factor;
		}
	}
	return reward;
}

double CFeatureRewardFunction::getReward(CState *oldState, CAction *action, CState *newState)
{
	double reward = 0;
	
	int type = oldState->getStateProperties()->getType() & (FEATURESTATE | DISCRETESTATE);
	switch (type)
	{
		case FEATURESTATE:
			{
				for (unsigned int oldS = 0; oldS < oldState->getNumDiscreteStates(); oldS++)
				{
					for (unsigned int newS = 0; newS < newState->getNumDiscreteStates(); newS++)
					{
						reward += getReward(oldState->getDiscreteState(oldS), action, newState->getDiscreteState(newS)) * oldState->getContinuousState(oldS) * newState->getContinuousState(newS);
					}
				}
				break;
			}
		case DISCRETESTATE:
			{
				reward = getReward(oldState->getDiscreteState(0), action, newState->getDiscreteState(0));
				break;
			}
		default:
			{
				reward = getReward(oldState->getDiscreteStateNumber(), action, newState->getDiscreteStateNumber());
			}
	}
	return reward;
}

CStateReward::CStateReward(CStateProperties *l_properties) : CStateObject(l_properties)
{
	this->properties = l_properties;
}

double CStateReward::getReward(CStateCollection *, CAction *, CStateCollection *newState)
{
	return this->getStateReward(newState->getState(properties));
}

CRewardFunctionFromValueFunction::CRewardFunctionFromValueFunction(CAbstractVFunction *vFunction, bool useNewState)
{
	this->vFunction = vFunction;
	this->useNewState = useNewState;
}

double CRewardFunctionFromValueFunction::getReward(CStateCollection *oldState, CAction *, CStateCollection *newState)
{
	double value = 0.0;
	if (useNewState)
	{
		value = vFunction->getValue(newState);
	}
	else
	{
		value = vFunction->getValue(oldState);
	}

	return value;
}


CFeatureRewardFunctionFromValueFunction::CFeatureRewardFunctionFromValueFunction(CStateModifier *discretizer, CFeatureVFunction *vFunction, bool useNewState) : CFeatureRewardFunction(discretizer)
{
	this->vFunction = vFunction;
	this->useNewState = useNewState;
}

CFeatureRewardFunctionFromValueFunction::~CFeatureRewardFunctionFromValueFunction()
{
}

double CFeatureRewardFunctionFromValueFunction::getReward(int oldState, CAction *, int newState)
{
	double value = 0.0;
	if (useNewState)
	{
		value = vFunction->getFeature(newState);
	}
	else
	{
		value = vFunction->getFeature(oldState);
	}

	return value;
}

