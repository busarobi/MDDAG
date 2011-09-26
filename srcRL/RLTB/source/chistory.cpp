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
#include "chistory.h"
#include "cstate.h"
#include "cenvironmentmodel.h"
#include "caction.h"
#include "cstatecollection.h"
#include "ril_debug.h"

#include <assert.h>


CStep::CStep(CStateProperties *properties, std::list<CStateModifier*> *modifiers, CActionSet *actions) : CStateObject(properties), CActionObject(actions)
{
	oldState = new CStateCollectionImpl(properties, modifiers);
	newState = new CStateCollectionImpl(properties, modifiers);

	action = NULL;

	actionData = new CActionDataSet(actions);
}

CStep::~CStep()
{
	delete oldState;
	delete newState;

	delete actionData;
}

CStepHistory::CStepHistory(CStateProperties *properties, CActionSet *actions) : CStateModifiersObject(properties), CActionObject(actions)
{
}



CBatchStepUpdate::CBatchStepUpdate(CSemiMDPListener *listener, CStepHistory *logger, int numUpdatesPerStep, int numUpdatesPerEpisode, std::list<CStateModifier *> *modifiers)
{
	this->listener = listener;
	this->steps = logger;
	this->numUpdates = numUpdates;

	addParameter("BatchStepUpdatesPerEpisode", numUpdatesPerEpisode);
	addParameter("BatchStepUpdatesPerStep", numUpdatesPerStep);

	step = new CStep(steps->getStateProperties(), modifiers, logger->getActions());
	dataSet = new CActionDataSet(logger->getActions());
}

CBatchStepUpdate::~CBatchStepUpdate()
{
	delete step;
	delete dataSet;
}

void CBatchStepUpdate::newEpisode()
{
	simulateSteps(listener, (int) getParameter("BatchStepUpdatesPerEpisode"));
}

void CBatchStepUpdate::nextStep(CStateCollection *, CAction *, CStateCollection *)
{
	simulateSteps(listener, (int) getParameter("BatchStepUpdatesPerStep"));
}

void CBatchStepUpdate::simulateAllSteps(CSemiMDPListener *listener)
{
	simulateSteps(listener, steps->getNumSteps());
}

void CBatchStepUpdate::simulateSteps(CSemiMDPListener *listener, int num)
{
	int rIndex = 0;
//	int sIndex = 0;
	int i = 0;

	CActionSet::iterator it = steps->getActions()->begin();
	for (; it != steps->getActions()->end(); it ++)
	{
		dataSet->setActionData(*it, (*it)->getActionData());
	}

	for (i = 0; i < num && i < steps->getNumSteps(); i++)
	{
		rIndex = rand() % steps->getNumSteps();

		steps->getStep(rIndex, step);

		listener->newEpisode();

		step->action->loadActionData(step->actionData->getActionData(step->action));
		listener->nextStep(step->oldState, step->action, step->newState);
	}

	it = steps->getActions()->begin();
	for (; it != steps->getActions()->end(); it ++)
	{
		(*it)->loadActionData(dataSet->getActionData(*it));
	}
}
