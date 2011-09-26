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
#include "cepisode.h"
#include "cstatecollection.h"
#include "caction.h"
#include "ril_debug.h"
#include "cstatemodifier.h"

#include <assert.h>

CEpisode::CEpisode(CStateProperties *properties, CActionSet *actions, bool autoNewEpisode) : CStateModifiersObject(properties), CStepHistory(properties, actions)
{
	this->autoNewEpisode = autoNewEpisode;

	stateCollectionList = new CStateCollectionList(properties);
	actionList = new CActionList(actions);
}

CEpisode::CEpisode(CStateProperties *properties, CActionSet *actions, std::list<CStateModifier *> *modifiers, bool autoNewEpisode) : CStateModifiersObject(properties), CStepHistory(properties,actions)
{
	this->autoNewEpisode = autoNewEpisode;

	stateCollectionList = new CStateCollectionList(properties);
	actionList = new CActionList(actions);

	addStateModifiers(modifiers);
}

CEpisode::~CEpisode()
{
	newEpisode();

	delete actionList;
	delete stateCollectionList;
}

void CEpisode::resetData()
{
	stateCollectionList->clearStateLists();

	actionList->clear();
}


void CEpisode::nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState )
{
	if (getNumSteps() == 0)
	{
		stateCollectionList->addStateCollection(oldState);
	}
	
	stateCollectionList->addStateCollection(newState);

	actionList->addAction(action);
}

void CEpisode::newEpisode()
{
	stateCollectionList->clearStateLists();

	actionList->clear();
}

int CEpisode::getNumSteps()
{
	return actionList->getSize();
}

void CEpisode::saveBIN(FILE *stream)
{
	stateCollectionList->saveBIN(stream);
	actionList->saveBIN(stream);
}

void CEpisode::loadBIN(FILE *stream)
{
	stateCollectionList->loadBIN(stream);
	actionList->loadBIN(stream);
}

void CEpisode::saveData(FILE *stream)
{
	stateCollectionList->saveASCII(stream);
	actionList->saveASCII(stream);

}

void CEpisode::loadData(FILE *stream)
{
	stateCollectionList->loadASCII(stream);
	actionList->loadASCII(stream);
}

void CEpisode::getStep(int num, CStep *step)
{
	assert(num < getNumSteps());

	stateCollectionList->getStateCollection(num, step->oldState);
	stateCollectionList->getStateCollection(num + 1, step->newState);

	step->action = actionList->getAction(num, step->actionData);
}

void CEpisode::getStateCollection(int index, CStateCollectionImpl *stateCollection)
{
	stateCollectionList->getStateCollection(index, stateCollection);
		
}

void CEpisode::getState(int index, CState *state)
{
	stateCollectionList->getState(index, state);
}

CStateList *CEpisode::getStateList(CStateProperties *properties)
{
	return stateCollectionList->getStateList(properties);
}

void CEpisode::addStateModifier(CStateModifier *modifier)
{
	stateCollectionList->addStateModifier(modifier);
	CStateModifiersObject::addStateModifier(modifier);
}

void CEpisode::removeStateModifier(CStateModifier *modifier)
{
	stateCollectionList->removeStateModifier(modifier);
	CStateModifiersObject::removeStateModifier(modifier);
}

CAction *CEpisode::getAction(unsigned int num, CActionDataSet *dataSet)
{
	return actionList->getAction(num, dataSet);
}

int CEpisode::getNumStateCollections() 
{
	return stateCollectionList->getNumStateCollections();
}


