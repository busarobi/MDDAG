// Copyright (C) 200
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

#include "cagent.h"
#include "cepisode.h"
#include "ril_debug.h"
#include "ccontinuousactions.h"
#include "cenvironmentmodel.h"
#include "cagentlistener.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "ril_debug.h"
#include <assert.h>

#ifdef WIN32

#include <conio.h>

bool RIL_Toolbox_KeyboardHit()
{
	bool result = _kbhit() != 0;
	if (result)
	{
		while (_kbhit() != 0) _getch();
	}
	return result;
}

void RIL_Toolbox_Set_Keypress()
{
}

void RIL_Toolbox_Reset_Keypress()
{
}

#else // UNIX

#include <poll.h>
#include <termios.h>
#include <unistd.h>

static struct termios RIL_Toolbox_stored_settings;

void RIL_Toolbox_Set_Keypress()
{
    struct termios new_settings;

    tcgetattr(0, &RIL_Toolbox_stored_settings);
    new_settings = RIL_Toolbox_stored_settings;
    /* Disable canonical mode, and set buffer size to 1 byte */
    new_settings.c_lflag &= (~ICANON);
    new_settings.c_cc[VTIME] = 0;
    new_settings.c_cc[VMIN] = 1;
    tcsetattr(0, TCSANOW, &new_settings);
    return;
}

void RIL_Toolbox_Reset_Keypress()
{
    tcsetattr(0, TCSANOW, &RIL_Toolbox_stored_settings);
    return;
}

bool RIL_Toolbox_KeyboardHit()
{
	pollfd p;
	p.fd = STDIN_FILENO;
    p.events = POLLIN;
    int numfds = poll(&p, 1, 1);
	return (numfds && p.revents);
}

#endif // WIN32

CSemiMDPSender::CSemiMDPSender()
{
	SMDPListeners = new std::list<CSemiMDPListener *>();
}

CSemiMDPSender::~CSemiMDPSender()
{
	delete SMDPListeners;
}

void CSemiMDPSender::addSemiMDPListener(CSemiMDPListener *listener)
{
	if (!isListenerAdded(listener))
	{
		SMDPListeners->push_back(listener);
	}
}

void CSemiMDPSender::removeSemiMDPListener(CSemiMDPListener *listener)
{
	SMDPListeners->remove(listener);
}

bool CSemiMDPSender::isListenerAdded(CSemiMDPListener *listener)
{
	for (std::list<CSemiMDPListener *>::iterator it = SMDPListeners->begin(); it != SMDPListeners->end(); it++) 
	{
		if ((*it) == listener)
		{
			return true;
		}
	}
	return false;
}

void CSemiMDPSender::startNewEpisode()
{
	for (std::list<CSemiMDPListener *>::iterator it = SMDPListeners->begin(); it != SMDPListeners->end(); it++) 
	{
		if ((*it)->enabled)
		{
			(*it)->newEpisode();
		}
	}
}

void CSemiMDPSender::sendNextStep(CStateCollection *lastState, CAction *action, CStateCollection *currentState)
{
	int i = 0;
	clock_t ticks1, ticks2;

	for (std::list<CSemiMDPListener *>::iterator it = SMDPListeners->begin(); it != SMDPListeners->end(); it++, i++) 
	{
	   if ((*it)->enabled)
	   {
			ticks1 = clock();
			(*it)->nextStep(lastState, action, currentState);
			ticks2 = clock();
			DebugPrint('t', "Time needed for listener %d: %d, %d\n", i,ticks2-ticks1, *it);
	   }
	}
}

void CSemiMDPSender::sendIntermediateStep(CStateCollection *lastState, CAction *action, CStateCollection *currentState)
{
	for (std::list<CSemiMDPListener *>::iterator it = SMDPListeners->begin(); it != SMDPListeners->end(); it++) 
	{
	   (*it)->intermediateStep(lastState, action, currentState);
	}
}


CSemiMarkovDecisionProcess::CSemiMarkovDecisionProcess() : CDeterministicController(new CActionSet()) 
{
	this->lastAction = NULL;

	currentSteps = 0;
	currentEpisodeNumber = 0;

	totalSteps = 0;
}

CSemiMarkovDecisionProcess::~CSemiMarkovDecisionProcess()
{
	delete actions;
}

/*
For the intermediate steps within an Extendedaction all the States occured while the ExtendedAction hasn't been finished, are also send with as the tuple 
Intermediate_State-Action-current_State. The duration of the Extendedaction gets also reduced in the intermediate Steps. 
*/
/** When the given action is finished (only MultiStepAction has the ability to be not finished) the step is sended to al Listeners. The Method also updates currentSteps.
@see CSemiMDPListener
*/
void CSemiMarkovDecisionProcess::sendNextStep(CStateCollection *lastState, CAction *action, CStateCollection *currentState)
{
	currentSteps++;
	totalSteps ++;

	bool finished = true;
	int duration = 1;

	// Action has finished ?
	if (action->isType(MULTISTEPACTION))
	{
		CMultiStepActionData *multiAction = dynamic_cast<CMultiStepAction *>(action)->getMultiStepActionData();
		finished = multiAction->finished;
		// get Duration
		duration = multiAction->duration;

		if (action->isType(PRIMITIVEACTION))
		{
			// if there was a multistep-primitiv action, the intermediate steps hasn't been
			// recognized, so update currentSteps
			currentSteps += duration - 1;
		}
	}

	if (finished)
	{
		CDeterministicController::nextStep(lastState, action, currentState);	

		// No ExtendedAction, send normal Step
		CSemiMDPSender::sendNextStep(lastState, action, currentState);
	}
}


CAction* CSemiMarkovDecisionProcess::getLastAction()
{
	return lastAction;
}

void CSemiMarkovDecisionProcess::addActions(CActionSet *actions)
{
	CActionSet::iterator it = actions->begin();

	for (; it != actions->end(); it  ++)
	{
		addAction(*it);

	}

}

void CSemiMarkovDecisionProcess::startNewEpisode()
{
	CDeterministicController::newEpisode();
	
	CSemiMDPSender::startNewEpisode();

	currentSteps = 0;
	currentEpisodeNumber ++;

	isFirstStep = true;

}
	

void CSemiMarkovDecisionProcess::addAction(CAction *action)
{
	actions->add(action);
	actionDataSet->addActionData(action);
}

CHierarchicalSemiMarkovDecisionProcess::CHierarchicalSemiMarkovDecisionProcess(CEpisode *loggedEpisode) : CSemiMarkovDecisionProcess(), CStateModifiersObject(loggedEpisode->getStateProperties())
{
	this->currentEpisode = loggedEpisode;
	pastState = new CStateCollectionImpl(currentEpisode->getStateProperties());
	currentState = new CStateCollectionImpl(currentEpisode->getStateProperties());

	addStateModifiers(currentEpisode->getStateModifiers());
}

CHierarchicalSemiMarkovDecisionProcess::CHierarchicalSemiMarkovDecisionProcess(CStateProperties *modelProperties, std::list<CStateModifier *> *modifiers) :CSemiMarkovDecisionProcess(), CStateModifiersObject(modelProperties)
{
	this->currentEpisode = NULL;

	pastState = new CStateCollectionImpl(modelProperties);
	currentState = new CStateCollectionImpl(modelProperties);

	if (modifiers)
	{
		addStateModifiers(modifiers);
	}
}


CHierarchicalSemiMarkovDecisionProcess::~CHierarchicalSemiMarkovDecisionProcess()
{
	delete pastState;
	delete currentState;
}

void CHierarchicalSemiMarkovDecisionProcess::addStateModifier(CStateModifier *modifier)
{
	pastState->addStateModifier(modifier);
	currentState->addStateModifier(modifier);

	CStateModifiersObject::addStateModifier(modifier);
}

void CHierarchicalSemiMarkovDecisionProcess::removeStateModifier(CStateModifier *modifier)
{
	pastState->removeStateModifier(modifier);
	currentState->removeStateModifier(modifier);

	CStateModifiersObject::removeStateModifier(modifier);
}

void CHierarchicalSemiMarkovDecisionProcess::sendNextStep(CAction *action)
{

	CDeterministicController::nextStep(pastState, action, currentState);	
	CSemiMarkovDecisionProcess::sendNextStep(pastState, action, currentState);


	if (action->isType(EXTENDEDACTION))
	{
		CExtendedAction *mAction = dynamic_cast<CExtendedAction *>(action);
		if (mAction->getMultiStepActionData()->finished && mAction->sendIntermediateSteps && currentEpisode != NULL)
		{
			// send the Intermediate Steps and the "double" Step of the ExtendedAction
		
			int oldDuration = mAction->getDuration();
			int episodeIndex = currentEpisode->getNumSteps() - 1;

			CAction *interAction = currentEpisode->getAction(episodeIndex);

			// set new duration of the extendedAction
			mAction->getMultiStepActionData()->duration = interAction->getDuration();

			// Send intermediate Steps
			if (mAction->sendIntermediateSteps)
			{
			
				interAction = currentEpisode->getAction(episodeIndex);

				// set new duration of the extendedAction
				mAction->getMultiStepActionData()->duration = interAction->getDuration();

				while (mAction->getMultiStepActionData()->duration < oldDuration)
				{
					assert(episodeIndex > 0);

					currentEpisode->getStateCollection(episodeIndex, pastState);
					CSemiMDPSender::sendIntermediateStep(pastState, mAction, currentState);	
					
					episodeIndex --;

					// set new duration of the extendedAction
					interAction = currentEpisode->getAction(episodeIndex);
					mAction->getMultiStepActionData()->duration += interAction->getDuration();
				}
			}
			
			assert(mAction->getDuration() == oldDuration);
		}
	}
}

void CHierarchicalSemiMarkovDecisionProcess::setLoggedEpisode(CEpisode *loggedEpisode)
{
	currentEpisode = loggedEpisode;
}

/** Sends the next Step if the Hierarchical SMDP has executed an action (i.e. he is in the hierarchical ActoinStack).
Before it sends the step, the executed Action from the SMDP is calculated and the new Tuple S-A-S is send to the Listeners.
*/
void CHierarchicalSemiMarkovDecisionProcess::nextStep(CStateCollection *oldState, CHierarchicalStack *actionStack, CStateCollection *newState)
{
	CAction *currentAction = getExecutedAction(actionStack);

	if (currentAction != NULL)
	{
		if (isFirstStep)
		{
			pastState->setStateCollection(oldState);
			isFirstStep = false;
		}
		bool sendStep = !currentAction->isType(EXTENDEDACTION);
		if (! sendStep)
		{
			CExtendedAction *eAction = dynamic_cast<CExtendedAction *>(currentAction);

			sendStep = eAction->getMultiStepActionData()->finished;
		}
		if (sendStep)
		{
			currentState->setStateCollection(newState);
			sendNextStep(currentAction);
			CStateCollectionImpl *buffer = pastState;
			pastState = currentState;
			currentState = buffer;
		}
	}
	if (multiStepData->finished && currentSteps > 0)
	{
		startNewEpisode();
	}
}


void CHierarchicalSemiMarkovDecisionProcess::newEpisode()
{
	startNewEpisode();
}

CAction* CHierarchicalSemiMarkovDecisionProcess::getNextHierarchyLevel(CStateCollection *state, CActionDataSet *actionDataSet)
{
	return getNextAction(state, actionDataSet);
}

/** Returns the action following the SMDP in the hierarchical Action Stack
*/
CAction *CHierarchicalSemiMarkovDecisionProcess::getExecutedAction(CHierarchicalStack *actionStack)
{
	CHierarchicalStack::iterator it = actionStack->begin();
	while (it != actionStack->end() && (*it) != this)
	{
		it ++;
	}
	if (it == actionStack->end())
	{
		return NULL;
	}
	else
	{
		it ++;
		assert(it != actionStack->end());
		return *it;
	}
}

/** Creates a new Agent. A Episode Object is also instantiated automatically and added to the ListenerList, logging can be turned of by
setLoggedEpisode(bool)*/
CAgent::CAgent(CEnvironmentModel *model) : CSemiMarkovDecisionProcess(), CStateModifiersObject(model->getStateProperties())
{
    this->model = model;
    lastAction = NULL;
    
    setParameters(1, 5000);
	keyboardBreaks = false;
	
	assert(model->getStateProperties());
//	printf("%d %d\n", model->getStateProperties()->getNumContinuousStates(), model->getStateProperties()->getNumDiscreteStates());

	currentEpisode = new CEpisode(model->getStateProperties(), actions);

	addSemiMDPListener(currentEpisode);
	bLogEpisode = true;

	currentState = new CStateCollectionImpl(model->getStateProperties());
	lastState = new CStateCollectionImpl(model->getStateProperties());

	modifiers = new std::list<CStateModifier *>;

	startNewEpisode();
}

CAgent::~CAgent()
{
	delete currentState;
	delete lastState;
	delete currentEpisode;
}

/** Tells the model which action to execute and then saves the new State. 
Send the oldState, the actoin and the newState as S-A-S Tuple to all Listeners (see CSemiMarkovDecisionProcess::sendNextStep(...)). 

There is a special treatment for actions of the type PRIMITIVEACTIONSTATECHANGE, @see CPrimitiveActionStateChanged  
*/
void CAgent::doAction(CAction *l_action) 
{
	CAction *action = l_action;
		
	

	int index = actions->getIndex(action);

	
	assert(action != NULL && action->isType(PRIMITIVEACTION) && index >= 0);

	if (model->isReset())
	{
		startNewEpisode();
	}

	if (isFirstStep)
	{
		isFirstStep = false;
		model->getState(currentState);
		currentState->newModelState();
		currentState->setResetState(model->isReset());

		if (DebugIsEnabled())
		{
			DebugPrint('+', "\nNew Episode (%d): ", this->getCurrentEpisodeNumber());
			DebugPrint('+', "start State: ");
			currentState->getState()->saveASCII(DebugGetFileHandle('+'));
			DebugPrint('+', "\n");
		}
	}

	CStateCollectionImpl *bufState = lastState;
	lastState = currentState;
	currentState = bufState;
	

	CPrimitiveAction* primAction = dynamic_cast<CPrimitiveAction*>(action);

	model->nextState(primAction);
	model->getState(currentState);

	currentState->setResetState(model->isReset());

	lastAction = action;

	if (DebugIsEnabled())
	{
		DebugPrint('+', "\nNew Step (%d): ", this->getCurrentStep());
		DebugPrint('+', "oldState: ");
		lastState->getState()->saveASCII(DebugGetFileHandle('+'));
		DebugPrint('+', "action: %d ", actions->getIndex(action));
		if (action->getActionData())
		{
			action->getActionData()->saveASCII(DebugGetFileHandle('+'));
		}
		DebugPrint('+', "currentState: ");
		currentState->getState()->saveASCII(DebugGetFileHandle('+'));
		DebugPrint('+', "\n");

	}

	sendNextStep(lastState, action, currentState);
}

void CAgent::setLogEpisode(bool bLogEpisode)
{
	if (this->bLogEpisode != bLogEpisode)
	{
		this->bLogEpisode = bLogEpisode;
		if (bLogEpisode)
		{
			SMDPListeners->push_front(currentEpisode);
		}
		else
		{
			removeSemiMDPListener(currentEpisode);
		}
	}
}

void CAgent::startNewEpisode()
{
	DebugPrint('a', "Starting new Episode\n");
	CSemiMarkovDecisionProcess::startNewEpisode();
	model->resetModel();
}

int CAgent::doControllerEpisode(int maxEpisodes, int maxSteps)
{
    setParameters(maxEpisodes, maxSteps);
	return doRun(false);
}

void CAgent::setParameters(int maxEpisodes, int maxSteps)
{
    this->maxEpisodes = maxEpisodes;
	this->maxSteps = maxSteps;
	this->currentEpisodeNumber = 0;
	this->currentSteps = 0;
}

int CAgent::doResume()
{
	return doRun(true);
}

int CAgent::doRun(bool )
{
	bool keyhit = false;
	RIL_Toolbox_Set_Keypress();
	while (currentEpisodeNumber < maxEpisodes  && !keyhit)
	{
		if (model->isReset() || currentSteps >= maxSteps)
		{
			startNewEpisode();
		}
		if (currentEpisodeNumber == 0)
		{
			currentEpisodeNumber ++;
		}

		while (!model->isReset() && currentSteps < maxSteps && (!keyhit)) 
		{        
            		doControllerStep();
			if (keyboardBreaks)
			{
				keyhit = RIL_Toolbox_KeyboardHit();
			}
        	}
			
    	}
	if (keyhit)
	{
		RIL_Toolbox_Reset_Keypress();
		return -1;
	}
	else
	{
		RIL_Toolbox_Reset_Keypress();
        	return currentSteps;
	}
}

void CAgent::doControllerStep()
{
	if (model->isReset())
	{
		startNewEpisode();
	}

	if (isFirstStep)
	{	
		isFirstStep = false;
		model->getState(currentState);
		currentState->newModelState();
		if (DebugIsEnabled())
		{
			DebugPrint('+', "\nNew Episode (%d): ", this->getCurrentEpisodeNumber());
			DebugPrint('+', "start State: ");
			currentState->getState()->saveASCII(DebugGetFileHandle('+'));
			DebugPrint('+', "\n");
		}
	}

	CAction *action = getNextAction(currentState);

	assert(action != NULL);

	action->loadActionData(actionDataSet->getActionData(action));

	doAction(action);
}

void CAgent::setKeyboardBreak(bool keyboardBreaks)
{
	this->keyboardBreaks = keyboardBreaks;
}

bool CAgent::getKeyboardBreak()
{
	return keyboardBreaks;
}

void CAgent::addAction(CPrimitiveAction *action)
{
	CSemiMarkovDecisionProcess::addAction(action);
}

void CAgent::addStateModifier(CStateModifier *modifier)
{
	lastState->addStateModifier(modifier);
	currentState->addStateModifier(modifier);

	currentEpisode->addStateModifier(modifier);

	CStateModifiersObject::addStateModifier(modifier);
}

void CAgent::removeStateModifier(CStateModifier *modifier)
{
	lastState->removeStateModifier(modifier);
	currentState->removeStateModifier(modifier);

	currentEpisode->removeStateModifier(modifier);

	CStateModifiersObject::removeStateModifier(modifier);
}

CEpisode *CAgent::getCurrentEpisode()
{
	return currentEpisode;
}

CStateCollection *CAgent::getCurrentState()
{
	if (isFirstStep)
	{	
		isFirstStep = false;
		model->getState(currentState);
		currentState->newModelState();
		if (DebugIsEnabled())
		{
			DebugPrint('+', "\nNew Episode (%d): ", this->getCurrentEpisodeNumber());
			DebugPrint('+', "start State: ");
			currentState->getState()->saveASCII(DebugGetFileHandle('+'));
			DebugPrint('+', "\n");
		}
	}

	return currentState;
}

CEnvironmentModel *CAgent::getEnvironmentModel()
{
	return model;
}

CHiearchicalAgent::CHiearchicalAgent(CAgent *l_agent, CHierarchicalSemiMarkovDecisionProcess *l_hierarchicSMDP) : CAgent(l_agent->getEnvironmentModel())
{
	realAgent = l_agent;
	hierarchicSMDP = l_hierarchicSMDP;

	setLogEpisode(false);
}

CHiearchicalAgent::~CHiearchicalAgent()
{

}

void CHiearchicalAgent::doAction(CAction *action)
{
	lastState->setStateCollection( realAgent->getCurrentState());

//	printf("HierarchicalAction: %d %d\n", realAgent->getActions()->getIndex( action), action);

//	printf("OldState: ");
//	lastState->getState()->saveASCII( stdout);
//	printf("\n");

	hierarchicSMDP->setNextAction( action);

	CMultiStepAction *multAction = NULL;
	
	if (action->isType(MULTISTEPACTION))
	{
		multAction = dynamic_cast<CMultiStepAction *>(action);
		multAction->getMultiStepActionData()->finished = false;
	
//		printf("MultstepAction (%d).. ", hierarchicSMDP->getActions()->getIndex( action));
	}	
	bool isFinished = false;

	
	do
	{
		realAgent->doControllerStep();
		isFinished = true;
		
		if (multAction)
		{
			isFinished = multAction->getMultiStepActionData()->finished;
			//printf("%d ", multAction->getMultiStepActionData()->duration);
		}
	}
	while (!isFinished);

	if (multAction)
	{
//		printf("\n");
	}

	currentState->setStateCollection( realAgent->getCurrentState());

//	printf("NewState: ");
//	currentState->getState()->saveASCII( stdout);
//	printf("\n");

	sendNextStep(lastState, action, currentState);
}

void CHiearchicalAgent::startNewEpisode()
{
	realAgent->startNewEpisode();
	CAgent::startNewEpisode();
}
