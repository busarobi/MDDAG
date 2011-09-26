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
#include "chierarchiccontroller.h"

#include "caction.h"
#include "cstatecollection.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "cutility.h"



#include <assert.h>

CHierarchicalStackSender::CHierarchicalStackSender()
{
	stackListeners = new std::list<CHierarchicalStackListener *>();
}

CHierarchicalStackSender::~CHierarchicalStackSender()
{
	delete stackListeners;
}

void CHierarchicalStackSender::addHierarchicalStackListener(CHierarchicalStackListener *listener)
{
	stackListeners->push_back(listener);
}

void CHierarchicalStackSender::removeHierarchicalStackListener(CHierarchicalStackListener *listener)
{
	stackListeners->remove(listener);
}

void CHierarchicalStackSender::startNewEpisode()
{
	for (std::list<CHierarchicalStackListener *>::iterator it = stackListeners->begin(); it != stackListeners->end(); it++) 
	{
	   (*it)->newEpisode();
	}
}

void CHierarchicalStackSender::sendNextStep(CStateCollection *oldState, CHierarchicalStack *actionStack, CStateCollection *newState)
{
	for (std::list<CHierarchicalStackListener *>::iterator it = stackListeners->begin(); it != stackListeners->end(); it++) 
	{
	   (*it)->nextStep(oldState, actionStack, newState);
	}
}

CHierarchicalStackEpisode::CHierarchicalStackEpisode(CActionSet *behaviors)
{
	actionStacks = new std::vector<CActionList *>();
	this->behaviors = behaviors;
}

CHierarchicalStackEpisode::~CHierarchicalStackEpisode()
{
	newEpisode();

	std::vector<CActionList *>::iterator it;
	for (it = actionStacks->begin(); it != actionStacks->end(); it ++)
	{
		delete *it;
	}
	delete actionStacks;
}

void CHierarchicalStackEpisode::nextStep(CHierarchicalStack *actionStack)
{
	CHierarchicalStack::iterator it;
	CActionList *stack = new CActionList(behaviors);
	for (it = actionStack->begin(); it != actionStack->end(); it ++)
	{
		stack->addAction(*it);
	}
	actionStacks->push_back(stack);
}

void CHierarchicalStackEpisode::newEpisode()
{
	for (unsigned int i = 0; i < actionStacks->size(); i++)
	{
		delete (*actionStacks)[i];
	}
}

void CHierarchicalStackEpisode::loadASCII(FILE *stream)
{
	unsigned int bufElems;
	int buf;
	fscanf(stream, "HierarchicalStacks: %d\n", &bufElems);

	for (unsigned int i = 0; i < bufElems; i++)
	{
		CActionList *stack = new CActionList(behaviors);
		fscanf(stream, "%d: ", &buf);
		stack->loadASCII(stream);
		fscanf(stream, "\n");

		actionStacks->push_back(stack);
	}
	
}

void CHierarchicalStackEpisode::saveASCII(FILE *stream)
{
	fprintf(stream, "HierarchicalStacks: %d\n", actionStacks->size());

	for (unsigned int i = 0; i < actionStacks->size(); i++)
	{
		fprintf(stream, "%d :", i);
		(*actionStacks)[i]->saveASCII(stream);
		fprintf(stream, "\n");
	}
}

void CHierarchicalStackEpisode::getHierarchicalStack(unsigned int index, CHierarchicalStack *actionStack, bool clearStack)
{
	if (clearStack)
	{
		actionStack->clear();
    }

	for (unsigned int i = 0; i < (*actionStacks)[index]->getNumActions(); i++)
	{
		actionStack->push_back(((*actionStacks)[index])->getAction(i, NULL));
	}

}

int CHierarchicalStackEpisode::getNumSteps()
{
	return actionStacks->size();
}



CHierarchicalController::CHierarchicalController(CActionSet *agentActions, CActionSet *allActions, CExtendedAction *rootAction) : CAgentController(agentActions)
{
	this->agentActions = agentActions;
	this->rootAction = rootAction;

	actionStack = new CHierarchicalStack();

	addParameter("MaxHierarchicExecution", 0);

	this->hierarchichActionDataSet = new CActionDataSet(allActions);
}

CHierarchicalController::~CHierarchicalController()
{
	delete actionStack;
	delete hierarchichActionDataSet;
}

int CHierarchicalController::getMaxHierarchicalExecution()
{
	return my_round(getParameter("MaxHierarchicExecution"));
}

void CHierarchicalController::setMaxHierarchicalExecution(int maxExec)
{
	setParameter("MaxHierarchicExecution", maxExec);
}

CAction* CHierarchicalController::getAgentAction(CHierarchicalStack *stack, CActionDataSet *actionDataSet)
{
	// returns primitiv action if it is member of the actionset of the controller
	CPrimitiveAction *agentAction = NULL;
	CPrimitiveAction *primitiveAction = dynamic_cast<CPrimitiveAction *>(*stack->rbegin());
	if (agentActions->getIndex(primitiveAction) >= 0)
	{
		agentAction = primitiveAction;
	}

	assert(agentAction != NULL);
	actionDataSet->setActionData(agentAction, hierarchichActionDataSet->getActionData(agentAction));
	return agentAction;
}

CAction *CHierarchicalController::getNextAction(CStateCollection *state, CActionDataSet *actionDataSet)
{
	actionStack->clear();
	rootAction->getHierarchicalStack(actionStack);

	CAction *stackElem = *actionStack->rbegin();
	CAction *nextElem = NULL;
	
	// rebuild the stack as long as a primitiv action occurs
    while (stackElem->isType(EXTENDEDACTION))
	{
		CExtendedAction *exAction = dynamic_cast<CExtendedAction *>(stackElem);
		nextElem = exAction->getNextHierarchyLevel(state, hierarchichActionDataSet);
		exAction->nextHierarchyLevel = nextElem;
		stackElem = nextElem;
		if (stackElem->isType(MULTISTEPACTION))
		{
			dynamic_cast<CMultiStepAction *> (stackElem)->getMultiStepActionData()->duration = 0;
		}
		if (stackElem->isType(EXTENDEDACTION))
		{
			dynamic_cast<CExtendedAction *> (stackElem)->nextHierarchyLevel = NULL;
		}
		actionStack->push_back(stackElem);
	}

	return getAgentAction(actionStack, actionDataSet);	
}

void CHierarchicalController::newEpisode()
{
	actionStack->clear();
	rootAction->getHierarchicalStack(actionStack);

	CHierarchicalStack::iterator it = actionStack->begin();
	it ++;

	CHierarchicalStackSender::startNewEpisode();
	
	for (; it != actionStack->end(); it++)
	{
		if ((*it)->isType(MULTISTEPACTION))
		{
			dynamic_cast<CMultiStepAction *>(*it)->getMultiStepActionData()->finished = false;
		}
	}

	for (CHierarchicalStack::iterator it = actionStack->begin(); it != actionStack->end(); it++)
	{
		if ((*it)->isType(MULTISTEPACTION))
		{
			 dynamic_cast<CMultiStepAction *>(*it)->getMultiStepActionData()->duration = 0;
		}
		if ((*it)->isType(EXTENDEDACTION))
		{
			dynamic_cast<CExtendedAction *> (*it)->nextHierarchyLevel = NULL;
		}
	}
	actionStack->clear();

}

void CHierarchicalController::nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState)
{
	bool finishedBefore = false;
	CMultiStepAction *newAction;
	int duration = 1;
	
	if (actionStack->size() == 0)
	{
		// Hierarchic stack wasnt build correctly, so the hierarchical controller is not the controller of the agent
		return;
	}
	// get duration of primitiv action
	
	if (action->isType(MULTISTEPACTION))
	{
		duration = dynamic_cast<CMultiStepAction *>(action)->getDuration();
	}
	
	bool finishedEpisode = newState->isResetState();
	
	for (CHierarchicalStack::iterator it = actionStack->begin(); it != actionStack->end(); it++)
	{
		if ((*it)->isType(MULTISTEPACTION))
		{
			newAction = dynamic_cast<CMultiStepAction *>(*it);
			// update all durations with the duration from primitiv action
			newAction->getMultiStepActionData()->duration += duration;
			
			
			if (finishedEpisode) // && (*it) != rootAction)
			{
				newAction->getMultiStepActionData()->finished = true;
			}
			else
			{
				// calculate the finished flag
				newAction->getMultiStepActionData()->finished =  finishedBefore || newAction->isFinished(oldState, newState);	
			}
			
			if ((*it) != rootAction && getParameter("MaxHierarchicExecution") > 0 && duration >= getParameter("MaxHierarchicExecution") )
			{
				newAction->getMultiStepActionData()->finished = true;
			}
			// if an element was finished, all other elements get finished too.
			finishedBefore = newAction->getMultiStepActionData()->finished;
		}
	}

	CHierarchicalStackSender::sendNextStep(oldState, actionStack, newState);

	/// delete all finished actions from the stack

	bool isFinished = !(*actionStack->rbegin())->isType(MULTISTEPACTION);
	if (! isFinished)
	{	
		newAction = dynamic_cast<CMultiStepAction *>(*actionStack->rbegin());
		isFinished = (newAction)->getMultiStepActionData()->finished;
	}

	CAction *lastAction = NULL;

	while (isFinished && actionStack->size() > 0)
	{
		lastAction = *actionStack->rbegin();
		
		/// set the next action duration

		if (lastAction->isType(MULTISTEPACTION))
		{
			dynamic_cast<CMultiStepAction *>(lastAction)->getMultiStepActionData()->duration = 0;
		}

		actionStack->pop_back();
		
		if (actionStack->size() > 0)
		{
			lastAction = *actionStack->rbegin();
			
			/// set the next action field of the last action.
			if (lastAction->isType(EXTENDEDACTION))
			{
				CExtendedAction *exAction = dynamic_cast<CExtendedAction *>(lastAction);
				exAction->nextHierarchyLevel = NULL;
			}
	
			isFinished = !lastAction->isType(MULTISTEPACTION);
			if (!isFinished)
			{	
				newAction = dynamic_cast<CMultiStepAction *>(lastAction);
				isFinished = (newAction)->getMultiStepActionData()->finished;
			}
		}
	}
	actionStack->clear();
}

void CHierarchicalController::intermediateStep(CStateCollection *oldState, CAction *action, CStateCollection *newState)
{
	nextStep(oldState, action, newState);
}

