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

#include <assert.h>
#include "ril_debug.h"
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "cstatemodifier.h"

CStateCollectionImpl::CStateCollectionImpl(CStateProperties *modelProperties) : CStateModifiersObject(modelProperties)
{
	modifiedStates = new std::map<CStateModifier *, CState *>();

	modelState = new CState(modelProperties);

	stateCalculated = new std::map<CStateModifier *, bool>();
}

CStateCollectionImpl::CStateCollectionImpl(CStateCollectionImpl *stateCol) : CStateModifiersObject(stateCol->getState()->getStateProperties())
{
	modifiedStates = new std::map<CStateModifier *, CState *>();

	modelState = new CState(stateCol->getState());
	
	stateCalculated = new std::map<CStateModifier *, bool>();

	addStateModifiers(stateCol->getStateModifiers());
}

CStateCollectionImpl::CStateCollectionImpl(CStateProperties *properties, std::list<CStateModifier *> *modifiers) : CStateModifiersObject(properties)
{
	modifiedStates = new std::map<CStateModifier *, CState *>();

	modelState = new CState(properties);

	stateCalculated = new std::map<CStateModifier *, bool>();

	addStateModifiers(modifiers);
}

CStateCollectionImpl::~CStateCollectionImpl()
{
	std::map<CStateModifier *, CState *>::iterator it;
	for (it = modifiedStates->begin(); it != modifiedStates->end(); it ++)
	{
		if ((*it).first->changeState)
		{
			(*it).first->removeStateCollection(this);
		}
		delete (*it).second;
	}
	delete modifiedStates;
	delete modelState;
	delete stateCalculated;
}

void CStateCollectionImpl::setStateCollection(CStateCollection *stateCollection)
{
	modelState->setState(stateCollection->getState(this->modelState->getStateProperties()));
	newModelState();

	std::map<CStateModifier *, CState *>::iterator it = modifiedStates->begin();
	for (; it != modifiedStates->end(); it++)
	{
		if (stateCollection->isMember((*it).first))
		{
			CState *targetState = (*it).second;
			targetState->setState(stateCollection->getState((*it).first));
			(*stateCalculated)[(*it).first] = true;
		}
	}
	setResetState(stateCollection->isResetState());
}

void CStateCollectionImpl::setResetState(bool reset)
{
	modelState->setResetState(reset);

	std::map<CStateModifier *, CState *>::iterator it = modifiedStates->begin();
	for (; it != modifiedStates->end(); it++)
	{
		CState *targetState = (*it).second;
		targetState->setResetState(reset);
	}
	CStateCollection::setResetState(reset);
}

void CStateCollectionImpl::newModelState()
{
	std::map<CStateModifier *, bool>::iterator it;
	for (it = stateCalculated->begin(); it != stateCalculated->end(); it ++)
	{
		(*it).second = false;
	}
}

void CStateCollectionImpl::calculateModifiedStates()
{
	std::map<CStateModifier *, CState *>::iterator it;
	std::map<CStateModifier *, bool>::iterator itCalc;

	for (itCalc = stateCalculated->begin(),it = modifiedStates->begin(); it != modifiedStates->end(); it ++, itCalc++)
	{
		if (!(*itCalc).second)
		{
			(*it).first->getModifiedState(this, (*it).second);
			(*itCalc).second = true;
		}
	}
}

void CStateCollectionImpl::setState(CState *state)
{
	if (state->getStateProperties() == modelState->getStateProperties())
	{
		modelState->setState(state);
		newModelState();
	}
	else
	{
		std::map<CStateModifier *, CState *>::iterator it = modifiedStates->find((CStateModifier *) state->getStateProperties());
		std::map<CStateModifier *, bool>::iterator itCalc = stateCalculated->find((CStateModifier *) state->getStateProperties());
	
	
		if (it != modifiedStates->end())
		{
			CState *targetState = (*it).second;
			targetState->setState(state);
			(*itCalc).second = true;
		}
	}
}

CState *CStateCollectionImpl::getState(CStateProperties *properties)
{
	if (properties == NULL || properties == modelState->getStateProperties())
	{
		return modelState;
	}
	CState *targetState = NULL;
	std::map<CStateModifier *, CState *>::iterator it = modifiedStates->find((CStateModifier *)properties);
	std::map<CStateModifier *, bool>::iterator itCalc = stateCalculated->find((CStateModifier *)properties);

	if (it != modifiedStates->end())
	{
		targetState = (*it).second;
		if (!(*itCalc).second)
		{
			(*it).first->getModifiedState(this, targetState);
			(*itCalc).second = true;
		}
	}
	assert(targetState != NULL);
	return targetState;
}

CState *CStateCollectionImpl::getState()
{
	return modelState;
}


void CStateCollectionImpl::addStateModifier(CStateModifier *modifier)
{
	CStateModifiersObject::addStateModifier(modifier);
	modifiedStates->insert(std::pair<CStateModifier *, CState *>(modifier, new CState(modifier)));
	stateCalculated->insert(std::pair<CStateModifier *, bool>(modifier, false));

	if (modifier->changeState)
	{
		modifier->registerStateCollection(this);
	}
}

void CStateCollectionImpl::removeStateModifier(CStateModifier *modifier)
{
	CStateModifiersObject::removeStateModifier(modifier);
	std::map<CStateModifier *, CState *>::iterator it = modifiedStates->find(modifier);
	std::map<CStateModifier *, bool>::iterator itCalc = stateCalculated->find(modifier);

	if (it != modifiedStates->end())
	{
		delete(*it).second;
        modifiedStates->erase(it);
		stateCalculated->erase(itCalc);
	}
}

bool CStateCollectionImpl::isMember(CStateProperties *stateProperties)
{
	return modelState->getStateProperties() == stateProperties || modifiedStates->find((CStateModifier*) stateProperties) != modifiedStates->end();
}

bool CStateCollectionImpl::isStateCalculated(CStateModifier *modifier)
{
	return (*this->stateCalculated)[modifier];
}

void CStateCollectionImpl::setIsStateCalculated(CStateModifier *modifier, bool isCalculated)
{
	(*this->stateCalculated)[modifier] = isCalculated;
}

/*
CState *CStateCollectionImpl::returnStateForExternSetting(CStateProperties *modifier)
{
	stateCalculated[modifier] = true;
	return (*modifiedStates)[modifier];
}
*/


CStateCollectionList::CStateCollectionList(CStateProperties *model) : CStateModifiersObject(model)
{
	stateLists = new std::list<CStateList *>;
	stateLists->push_back(new CStateList(model));
}

CStateCollectionList::CStateCollectionList(CStateProperties *model, std::list<CStateModifier *> *modifiers) : CStateModifiersObject(model)
{
	stateLists = new std::list<CStateList *>;
	stateLists->push_back(new CStateList(model));

	addStateModifiers(modifiers);
}

CStateCollectionList::~CStateCollectionList()
{
	clearStateLists();
	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		delete (*it);
	}
	delete stateLists;
}

void CStateCollectionList::clearStateLists()
{
	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		(*it)->clear();
	}
	
}

void CStateCollectionList::addStateCollection(CStateCollection *stateCollection)
{
	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		(*it)->addState(stateCollection->getState((*it)->getStateProperties()));
	}
}

void CStateCollectionList::removeLastStateCollection()
{
	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		(*it)->removeLastState();
	}
}

void CStateCollectionList::getStateCollection(int index, CStateCollectionImpl *stateCollection)
{
	std::list<CStateList *>::iterator it = stateLists->begin();

	if (stateCollection->isMember((*it)->getStateProperties()))
	{
		(*it)->getState(index, stateCollection->getState((*it)->getStateProperties()));
	}
	stateCollection->newModelState();
	std::list<CStateModifier *>::iterator itMod = modifiers->begin();

	it ++;
	
	for (; it != stateLists->end(); it++, itMod++)
	{
		if (stateCollection->isMember((*it)->getStateProperties()))
		{
			(*it)->getState(index, stateCollection->getState(*itMod));
			stateCollection->setIsStateCalculated(*itMod, true);
		}
	}
	
	stateCollection->setResetState(stateCollection->getState()->isResetState());		
}

void CStateCollectionList::getState(int index, CState *state)
{
	state->resetState();

	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		if ((*it)->getStateProperties() == state->getStateProperties())
		{
			(*it)->getState(index, state);
		}
	}	
}

CStateList *CStateCollectionList::getStateList(CStateProperties *properties)
{
	std::list<CStateList *>::iterator it;
	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		if ((*it)->getStateProperties() == properties)
		{
			return *it;
		}
	}	
	return NULL;
}

void CStateCollectionList::addStateModifier(CStateModifier *modifier)
{
	CStateModifiersObject::addStateModifier(modifier);
	stateLists->push_back(new CStateList(modifier));
}

void CStateCollectionList::removeStateModifier(CStateModifier *modifier)
{
	CStateModifiersObject::removeStateModifier(modifier);
	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		if ((*it)->getStateProperties() == modifier)
		{
			(*it)->clear();
			delete (*it);
			
			stateLists->erase(it);
			break;
		}
	}
}

void CStateCollectionList::loadASCII(FILE *stream)
{
	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		(*it)->loadASCII(stream);
	}
}

void CStateCollectionList::saveASCII(FILE *stream)
{
	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		(*it)->saveASCII(stream);
	}
}
	
void CStateCollectionList::saveBIN(FILE *stream)
{
	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		(*it)->saveBIN(stream);
	}
}

void CStateCollectionList::loadBIN(FILE *stream)
{
	std::list<CStateList *>::iterator it;

	for (it = stateLists->begin(); it != stateLists->end(); it++)
	{
		(*it)->loadBIN(stream);
	}
}

int CStateCollectionList::getNumStateCollections()
{
	return (*stateLists->begin())->getNumStates();
}

