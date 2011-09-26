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

#include "cagentcontroller.h"
#include "ril_debug.h"
#include "cactionstatistics.h"
#include "caction.h"
#include <assert.h>


CAgentController::CAgentController(CActionSet *actions) : CActionObject(actions)
{
}

CAgentController::~CAgentController()
{
}

CAgentStatisticController::CAgentStatisticController(CActionSet *actions) : CAgentController(actions)
{
}

CAction* CAgentStatisticController::getNextAction(CStateCollection *state, CActionDataSet *data )
{
	return getNextAction(state, data, NULL);
}


CDeterministicController::CDeterministicController(CActionSet *actions) : CAgentController(actions)
{
	this->controller = NULL;
	this->statisticController = NULL;
	useStatisticController = false;
	statistics = new CActionStatistics();
	initStatistics();
	nextAction = NULL;

	actionDataSet = new CActionDataSet(getActions());
}

CDeterministicController::CDeterministicController(CAgentController *controller) : CAgentController(controller->getActions())
{
	this->controller = controller;
	this->statisticController = NULL;
	useStatisticController = false;
	statistics = new CActionStatistics();
	initStatistics();
	nextAction = NULL;

	actionDataSet = new CActionDataSet(getActions());
}

CDeterministicController::CDeterministicController(CAgentStatisticController *controller) : CAgentController(controller->getActions())
{
	this->statisticController = controller;
	this->controller = controller;
	useStatisticController = false;
	statistics = new CActionStatistics();
	initStatistics();
	nextAction = NULL;

	actionDataSet = new CActionDataSet(getActions());
}

CDeterministicController::~CDeterministicController()
{
	delete statistics;
	delete actionDataSet;
}

CAction* CDeterministicController::getNextAction(CStateCollection *state, CActionDataSet *data)
{
	if (nextAction == NULL)
	{
		if (useStatisticController)
		{
			nextAction = statisticController->getNextAction(state, actionDataSet, statistics);
		}
		else
		{
			if (controller != NULL)
			{
				nextAction = controller->getNextAction(state, actionDataSet);
			}
			else
			{
				nextAction = NULL;
			}
		}
	}
	if (data && nextAction)
	{
		data->setActionData(nextAction, actionDataSet->getActionData(nextAction));
	}
	return nextAction;
}

void CDeterministicController::nextStep(CStateCollection *, CAction *, CStateCollection *)
{
	nextAction = NULL;
}

void CDeterministicController::newEpisode()
{
	nextAction = NULL;
}

void CDeterministicController::initStatistics()
{
	assert (statistics != NULL);
	statistics->owner = this;
	statistics->action = NULL;
	statistics->equal = 0;
	statistics->superior = 0;
	statistics->probability = 0.0;
}

CActionStatistics *CDeterministicController::getLastActionStatistics()
{
	return statistics;
}

bool CDeterministicController::isUsingStatisticController()
{
	return useStatisticController;
}

void CDeterministicController::setController(CAgentController *controller)
{
	this->controller = controller;
	this->statisticController = NULL;
	useStatisticController = false;
	initStatistics();
}

void CDeterministicController::setController(CAgentStatisticController *controller)
{
	this->statisticController = controller;
	this->controller = controller;
	useStatisticController = false;
}

void CDeterministicController::setNextAction(CAction *action, CActionData *data)
{
	nextAction  = action;

	if (data)
	{
		actionDataSet->setActionData( nextAction, data);
	}
}
