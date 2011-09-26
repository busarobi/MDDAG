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

#ifndef C_Hierarchical_CONTROLLER
#define C_Hierarchical_CONTROLLER

#include "cagentcontroller.h" 
#include "cagentlistener.h"

class CStateCollection;
class CStateProperties;
class CState;
class CAction;
class CActionSet;
class CActionDataSet;
class CActionList;
class CExtendedAction;

#include <list>
#include <vector>

class CHierarchicalStack;

/// Listener who gets the Hierarchical Stack instead of an action
/** These listeners can only be added to the hierachic controller, they get the full information about 
the current hierarchical stack, which can also be used for learning. */
class CHierarchicalStackListener
{
public:
	virtual ~CHierarchicalStackListener() {};

	virtual void nextStep(CStateCollection *oldState, CHierarchicalStack *actionStack, CStateCollection *newState) = 0;
	virtual void newEpisode() {};
};

/// Base class for sending the hierarchical stack to the listeners
class CHierarchicalStackSender
{
protected:
	std::list<CHierarchicalStackListener *> *stackListeners;
public:
	CHierarchicalStackSender();
	virtual ~CHierarchicalStackSender();

	void addHierarchicalStackListener(CHierarchicalStackListener *listener);
	void removeHierarchicalStackListener(CHierarchicalStackListener *listener);

	virtual void startNewEpisode();
	virtual void sendNextStep(CStateCollection *oldState, CHierarchicalStack *actionStack, CStateCollection *newState);
};

/// Class for logging the Hierarchical Stack of a training trial
/** Stores each step the Hierarchical stack in an action List. Must be added as listener of a CHierarchicalController object.
*/
class CHierarchicalStackEpisode : public CHierarchicalStackListener
{
protected:
	std::vector<CActionList *> *actionStacks;
	CActionSet *behaviors;
public:
	CHierarchicalStackEpisode(CActionSet *behaviors);
	virtual ~CHierarchicalStackEpisode();

	virtual void nextStep(CHierarchicalStack *actionStack);
	virtual void newEpisode();

	void loadASCII(FILE *stream);
	void saveASCII(FILE *stream);

	void saveAction(int index, FILE *stream);

	void getHierarchicalStack(unsigned int index, CHierarchicalStack *actionStack, bool clearStack = true);
	virtual int getNumSteps();
};

/// Class for calculating a hierarchical execution of a hierarchical structure
/**
The hierarchical controller calculates a hierarchical policy, given the root element from an hierarchical learning structure.
The hierarchical learning structure is build by the Hierarchical Semi Markov Decision Processes. The hierarchical stores an hierarchical
action stack and executes each hierarchical action as long as it (or an action with higher hierarchy in the stack) is finished.
The hierarchical controller also manages the calculation of the duration and the finished flag of all extended actions. 
The controller also has a listener list of stack listeners (from superclass CHierarchicalStackSender). Each component which 
needs acces to the hierarchical stack needs to be a hierarchical stack listener (CHierarchicStackListener) and must be added
to the listener list of the controller. The controller, if added to the listener list of the again then sends each step the states 
(old and newstate) and the hierarchical stack to his listeners. So all hierarchical semi MDP's must be added as to the controller's listeners, even
seme MDP's which are not directly the root of the hierarchy.
\par
You can also set the duration of the hierarchical execution, so if an extended action has a already took longer than the maximum 
duration the extended action is finished by the controller. With this feature you can make a transition from
hierarchical to flat execution during learning. 
\par
The hierarchical controller always returns a primitiv action (the last action on the stack) from his getNextAction method for the agent to execute.
All primitiv actions returned from the hierarchical structure must be member of the controllers action set!
@see CHierarchicalStackListener
@see CHierarchicalSemiMarkovDecisionProcess.
*/

class CHierarchicalController : public CAgentController, public CHierarchicalStackSender, public CSemiMDPListener
{
protected:
/// The agent's actions
	CActionSet *agentActions;
/// The actual hierarchical stack
	CHierarchicalStack *actionStack;
/// The root of the hierarchical structure
	CExtendedAction *rootAction;

	CActionDataSet *hierarchichActionDataSet;

/// returns the action for the agent from the stack (the last action on stack)
	virtual CAction* getAgentAction(CHierarchicalStack *stack, CActionDataSet *actionDataSet);
public:
/// creates the hierarchical controller with the actionset he can choose from and the root of the hierarchical structure
/** All primitiv actions returned from the hierarchical structure must be member of the action set!*/
	CHierarchicalController(CActionSet *agentActions, CActionSet *allActions, CExtendedAction *rootAction);
	~CHierarchicalController();
/// get maximum duration of hierarchical execution
	int getMaxHierarchicalExecution();
/// set maximum duration of hierarchical execution
	void setMaxHierarchicalExecution(int maxExec);

/// returns the primitiv action from the action stack
/** Builds and renews missing parts of the action stack */
	virtual CAction *getNextAction(CStateCollection *state, CActionDataSet *actionDataSet);

/// sends the states and action stacks to the listeners.
/** Calculates the duration of the actions (adds the duratoin of the executed primitiv action to each 
extended action), and the finished flags. If one action is finished, all other actions with lower hierarchy in
the action stack get marked as finished too.
<p>
After that calculation it sends the action stack to all listeners and then it deletes the finished actions
from the stack. The missing stack elements are renewed by the method getAction. */

	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState);
/// only calls nextStep
	virtual void intermediateStep(CStateCollection *oldState, CAction *action, CStateCollection *newState);

	virtual void newEpisode();
};

#endif

