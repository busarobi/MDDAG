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

#include "caction.h"
#include "cstate.h"
#include "ril_debug.h"

#include <assert.h>
#include <math.h>

CActionData::CActionData()
{
	bIsChangeAble = true;
}

bool CActionData::isChangeAble()
{
	return bIsChangeAble;
}

void CActionData::setIsChangeAble(bool changeAble)
{
	this->bIsChangeAble = changeAble;
}

CAction::CAction()
{
	this->type = 0;
	actionData = NULL;
}

CAction::CAction(CActionData *actionData)
{
	this->type = 0;
	this->actionData = actionData;
}

CAction::~CAction()
{
	if (actionData)
	{
		delete actionData;
	}
}

int CAction::getType()
{
	return type;
}

void CAction::addType(int Type)
{
	type = type | Type;	
}

bool CAction::isType(int type)
{
	return (this->type & type) > 0;
}

void CAction::loadActionData(CActionData *actionData)
{
	if (this->actionData)
	{
		this->actionData->setData(actionData);
	}
}

bool CAction::equals(CAction *action)
{
	return this == action;
}

bool CAction::isSameAction(CAction *action, CActionData *)
{
	return this == action;
}


CActionData* CAction::getActionData()
{
	return actionData;
}

CActionData *CAction::getNewActionData()
{ 
	return NULL;
}

CMultiStepActionData::CMultiStepActionData()
{
	finished = true;
	duration = 1;
}

void CMultiStepActionData::loadASCII(FILE *stream)
{
	int buff;
	fscanf(stream, "[%d %d] ", &duration, &buff);
	finished = buff != 0;
}

void CMultiStepActionData::saveASCII(FILE *stream)
{
	fprintf(stream, "[%d %d] ", duration, finished);
}

void CMultiStepActionData::loadBIN(FILE *stream)
{
	fread(&duration, sizeof(int), 1, stream);
	fread(&finished, sizeof(bool), 1, stream);
}

void CMultiStepActionData::saveBIN(FILE *stream)
{
	fwrite(&duration, sizeof(int), 1, stream);
	fwrite(&finished, sizeof(bool), 1, stream);
}

void CMultiStepActionData::setData(CActionData *actionData)
{
	if (isChangeAble())
	{
		CMultiStepActionData *data = dynamic_cast<CMultiStepActionData *>(actionData);

		duration = data->duration;
		finished = data->finished;
	}
}

CMultiStepAction::CMultiStepAction() : CAction(new CMultiStepActionData())
{
	type = type | MULTISTEPACTION;

	this->multiStepData = dynamic_cast<CMultiStepActionData *>(actionData);
}

CMultiStepAction::CMultiStepAction(CMultiStepActionData *actionData) : CAction(actionData)
{
	type = type | MULTISTEPACTION;

	this->multiStepData = actionData;
}

CActionData *CMultiStepAction::getNewActionData()
{
	CMultiStepActionData *data = new CMultiStepActionData();
	return dynamic_cast<CActionData *>(data);
}


CPrimitiveAction::CPrimitiveAction(CMultiStepActionData *actionData) : CAction(actionData)
{
	this->type = this->type | PRIMITIVEACTION;
}


CPrimitiveAction::CPrimitiveAction()
{
	this->type = this->type | PRIMITIVEACTION;
}


CPrimitiveAction::~CPrimitiveAction()
{
	
}

CExtendedAction::CExtendedAction()
{
	type = type | EXTENDEDACTION;
	sendIntermediateSteps = true;
	nextHierarchyLevel = NULL;
}

CExtendedAction::CExtendedAction(CMultiStepActionData *actionData) : CMultiStepAction(actionData)
{
	type = type | EXTENDEDACTION;
	sendIntermediateSteps = true;
	nextHierarchyLevel = NULL;
}

void CExtendedAction::getHierarchicalStack(CHierarchicalStack *actionStack)
{
	actionStack->push_back(this);

	if (nextHierarchyLevel != NULL)
	{
		if (nextHierarchyLevel->isType(EXTENDEDACTION))
		{
			CExtendedAction *exAction = dynamic_cast<CExtendedAction *>(nextHierarchyLevel);
			exAction->getHierarchicalStack(actionStack);
		}
		else
		{
			actionStack->push_back(nextHierarchyLevel);
		}
	}
}

CHierarchicalStack::CHierarchicalStack()
{

}

CHierarchicalStack::~CHierarchicalStack()
{
}


void CHierarchicalStack::clearAndDelete()
{
	for (CHierarchicalStack::iterator it = begin(); it != end(); it ++)
	{
		delete (*it);
	}
	clear();
}


CActionSet::CActionSet()
{
}

CActionSet::~CActionSet()
{
}

int CActionSet::getIndex(CAction *action)
{
	int index = -1, i = 0;
	CActionSet::iterator it;

	assert(action != NULL);

	for (it = begin(); it != end(); it ++, i++)
	{
		if ((*it)->equals(action))
		{
			index = i;
			break;
		}
	}
	return index;
}

bool CActionSet::isMember(CAction *action)
{
	CActionSet::iterator it;

	assert(action != NULL);

	for (it = begin(); it != end(); it ++)
	{
		if ((*it)->equals(action))
		{
			return true;
		}
	}
	return false;
}

void CActionSet::add(CActionSet *actions)
{
	if (actions != NULL)
	{
		CActionSet::iterator it = actions->begin();

		for(; it != actions->end(); it++)
		{
			push_back(*it);
		}
	}
}

CAction *CActionSet::get(unsigned int index)
{
	assert(index < size());

	CActionSet::iterator it = begin();
	for (unsigned int i = 0; i < index; it++, i++);
    return(*it);
}

void CActionSet::add(CAction *action)
{
	push_back(action);
}

void CActionSet::getAvailableActions(CActionSet *availableActions, CStateCollection *stateCol)
{
	availableActions->clear();
	for (CActionSet::iterator it = begin(); it != end(); it++)
	{
		if ((*it)->isAvailable(stateCol))
		{
			availableActions->add(*it);
		}
	}
}

CActionList::CActionList(CActionSet *actions) : CActionObject(actions)
{
	actionIndices = new std::vector<int>();
	actionDatas = new std::map<int, CActionData *>();
}

CActionList::~CActionList()
{
	clear();

	delete actionIndices;
	delete actionDatas;
}

void CActionList::addAction(CAction *action)
{
	unsigned int numAction = actionIndices->size();
	actionIndices->push_back(actions->getIndex(action));
	CActionData *data = action->getNewActionData();
	if (data != NULL)
	{
		data->setData(action->getActionData());
		(*actionDatas)[numAction] = data;
	}
}

CAction *CActionList::getAction(unsigned int num, CActionDataSet *l_data)
{
	CAction *action = actions->get((*actionIndices)[num]);

	CActionData *data = (*actionDatas)[num];
	if (data != NULL && l_data != NULL) 
	{
		l_data->setActionData(action, data);
	}
	return action;
}

void CActionList::loadASCII(FILE *stream)
{
	int numActions = 0, bufAction = 0;
	fscanf(stream, "ActionList: %d Actions\n", &numActions);
	for (int i = 0; i < numActions;  i ++)
	{
		fscanf(stream, "%d ", &bufAction);
		actionIndices->push_back(bufAction);
		CActionData *data = actions->get(bufAction)->getNewActionData();
		if (data != NULL)
		{
			data->loadASCII(stream);
			(*actionDatas)[i] = data;
		}
	}
	fscanf(stream, "\n");
}

void CActionList::saveASCII(FILE *stream)
{
	fprintf(stream, "ActionList: %d Actions\n", getSize());
	for (unsigned int i = 0; i < getSize(); i++)
	{
		fprintf(stream, "%d ", (*actionIndices)[i]);
		if ((*actionDatas)[i] != NULL)
		{
			(*actionDatas)[i]->saveASCII(stream);
		}
	}
	fprintf(stream, "\n");
}


void CActionList::saveBIN(FILE *stream)
{
	int size = getSize();
	fwrite(&size, sizeof(int),1, stream);
	for (unsigned int i = 0; i < getSize(); i++)
	{
		fwrite(&(*actionIndices)[i], sizeof(int), 1, stream);
		if ((*actionDatas)[i] != NULL)
		{
			(*actionDatas)[i]->saveBIN(stream);
		}
	}
}

void CActionList::loadBIN(FILE *stream)
{
	int numActions = 0, bufAction = 0;
	fread(&numActions, sizeof(int), 1, stream);
	for (int i = 0; i < numActions;  i ++)
	{
		int r = fread(&bufAction, sizeof(int), 1, stream);
		assert(r == 1);
		actionIndices->push_back(bufAction);
		CActionData *data = actions->get(bufAction)->getNewActionData();
		if (data != NULL)
		{
			data->loadBIN(stream);
			(*actionDatas)[i] = data;
		}
	}
}


unsigned int CActionList::getSize()
{
	return actionIndices->size();
}

unsigned int CActionList::getNumActions()
{
	return actions->size();
}


void CActionList::clear()
{
	actionIndices->clear();
	std::map<int, CActionData *>::iterator it = actionDatas->begin();

	for (; it != actionDatas->end(); it++)
	{		
		delete ((*it).second);
	}
	actionDatas->clear();
}


CActionDataSet::CActionDataSet(CActionSet *actions) //: CActionObject(actions)
{
	actionDatas = new std::map<CAction *, CActionData *>();

	CActionSet::iterator it;
	for (it = actions->begin(); it != actions->end(); it++)
	{
		(*actionDatas)[(*it)] = (*it)->getNewActionData();
		if ((*actionDatas)[(*it)] != NULL)
		{
			(*actionDatas)[(*it)]->setData((*it)->getActionData());
		}
	}
}

CActionDataSet::CActionDataSet() //: CActionObject(actions)
{
	actionDatas = new std::map<CAction *, CActionData *>();

	
}

CActionDataSet::~CActionDataSet()
{
	std::map<CAction *, CActionData *>::iterator it;

	for (it = actionDatas->begin(); it != actionDatas->end(); it ++)
	{
		if ((*it).second != NULL)
		{
			delete (*it).second;
		}
	}
	delete actionDatas;
}

CActionData *CActionDataSet::getActionData(CAction *action)
{
	return (*actionDatas)[action];
}

void CActionDataSet::setActionData(CAction *action, CActionData *actionData)
{
	CActionData *data = (*actionDatas)[action];
	if (data)
	{
		data->setData(actionData);
	}
}

void CActionDataSet::addActionData(CAction *action)
{
	if ((*actionDatas)[action] == NULL)
	{
		(*actionDatas)[action] = action->getNewActionData();
	}
}

void CActionDataSet::removeActionData(CAction *action)
{
	if ((*actionDatas)[action] != NULL)
	{
		delete (*actionDatas)[action];
	}
	(*actionDatas)[action] = NULL;
}


