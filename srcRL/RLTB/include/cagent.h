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

#ifndef CAGENT_H
#define CAGENT_H

#include <list>


#include "cagentcontroller.h"
#include "caction.h"
#include "cbaseobjects.h"
#include "chierarchiccontroller.h"


class CSemiMDPListener;
class CEpisode;
class CEnvironmentModel;
class CStateCollection;
class CStateCollectionImpl;

/// Class for sending the State-Action-State Tuple to the Listeners
/** Maintains a List of CSemiMDPListeners. The class provides methods for sending the Listeners that a new
Episode has started, a new Step (State-Action-State Tuple) or an intermediate Step has occured.
@see CSemiMDPListener
*/
class CSemiMDPSender
{
protected:
	std::list<CSemiMDPListener *> *SMDPListeners;
public:
	CSemiMDPSender();
	virtual ~CSemiMDPSender();

/// Add a Listener to the Listener-List
	void addSemiMDPListener(CSemiMDPListener *listener);
/// Remove a Listener from the Listerner-List
	void removeSemiMDPListener(CSemiMDPListener *listener);

	bool isListenerAdded(CSemiMDPListener *listener);

/// Tells all Listeners that a new Episode has occured
	virtual void startNewEpisode();
/// Sends the State-Action-State Tuple to all Listeners
	virtual void sendNextStep(CStateCollection *lastState, CAction *Action,  CStateCollection *currentState);
/// Sends the State-Action-State Tuple to all Listeners, indicating that ist was an intermediate step
	virtual void sendIntermediateStep(CStateCollection *lastState, CAction *Action, CStateCollection *currentState);
};

/// Class for providing the general Functions for the learning Environment
/**
CSemiMarkovDecisionProcess is the super class of all "acting" agents. It maintains a list of available Actions for the 
SMDP. It also loggs the number of Episodes and the Number of Steps done in the current Epsiode. 
It provides the functionality for sending a Semi-Markov Step, but its only able to send PrimitiveActions. 
For extended Actions you have to use the hierarchicalMDP.
/par
The class is a subclass of CDeterministicController in order to make any Controller assigned to the SMDP a deterministic Controller. The CDeterministicController Object is always the first Object to be informed about the new Step, and its not in the ListenerList (Recursions).
This feature is needed for learning the exact Policy of the agent (see CSarsaLearner), so the agent can be used as estimation policy.
@see CAgent
@see CHierarchicalalSemiMarkovDecisionProcess
*/
class CSemiMarkovDecisionProcess : public CDeterministicController, public CSemiMDPSender
{
protected:

	CAction *lastAction;
	
	int currentEpisodeNumber;
	int currentSteps;

	int totalSteps;
	
	bool isFirstStep;


public:

	CSemiMarkovDecisionProcess();
	~CSemiMarkovDecisionProcess();

/// Sends the next Step to all Listeners. I.e that if the Action is a finished MultiStepAction.
	virtual void sendNextStep(CStateCollection *lastState, CAction *action, CStateCollection *currentState);

/// Returns the last Action sent to all Listeners
	CAction* getLastAction();

/// Sends to all Listeners that a new Episode occured and updates currentEpisodeNummer.
	virtual void startNewEpisode();

/// Returns the number of Episodes.
	int getCurrentEpisodeNumber() {return this->currentEpisodeNumber;};
/// Returns the number of steps.
	int getCurrentStep() {return this->currentSteps;};

	int getTotalSteps() {return this->totalSteps;};

/// Adds an Action to the ActionSet of the SMDP. 
	virtual void addAction(CAction *action);
	virtual void addActions(CActionSet *action);
};

/// Subclass of CSemiMarkovDecisionProcess, used for hierarchical Learning
/**
This abstract class provides full Hierarchical learning functionality. It implements the CHierarchicalStackListener interface, 
so the HierarchicalController can inform the SMDP about a hierarchical Step. Than the SMDP sends the State-Action-State Tuple
with the action done by the specific hierarchical SMDP. \par
In order to provide hierarchical Functionality the class also represents an ExtendedAction, so it can be used as Action for another hierarchical SMDP. 
It can't be used as action for the agent, you have to use CHierarchicalController to create the HierarchicalStructure.
Use the hierarchical controller as controller for the agent, the agent itself does'nt know anything about the hierarchical structure of the learning Problem. 
The class is abstract because the isFinished Method from CMultiStepAction remains to be implemented.
@see CHierarchicalController
*/
class CHierarchicalSemiMarkovDecisionProcess : public CSemiMarkovDecisionProcess, public CHierarchicalStackListener, public CExtendedAction, public CStateModifiersObject
{
protected:
/// Returns the action done by the SMDP 
	virtual CAction *getExecutedAction(CHierarchicalStack *actionStack);
	
	/** Pointer to the currentEpisode, the currenEpisode must be updated before the sendNextStep method is called. So currentEpisode 
	has to be the first Element of the agent's Listener-List. Needed for determining the intermediate Steps.
	*/
	CEpisode *currentEpisode;

	CStateCollectionImpl *pastState;
	CStateCollectionImpl *currentState;
public:
/**
Creates a new hierarchical SMDP. The episode is needed for reconstruction of the intermediate and hierarchical steps.
 @param currentEpisode Pointer to the current Episode. It is recommended to use the currentEpisode Object of the Agent.
*/
	CHierarchicalSemiMarkovDecisionProcess(CEpisode *currentEpisode);
	CHierarchicalSemiMarkovDecisionProcess(CStateProperties *modelProperties, std::list<CStateModifier *> *modifiers = NULL);

	~CHierarchicalSemiMarkovDecisionProcess();

	virtual void setLoggedEpisode(CEpisode *loggedEpisode);

	virtual void nextStep(CStateCollection *oldState, CHierarchicalStack *actionStack, CStateCollection *newState);
	virtual void newEpisode();

/// Sends the nextStep to the listeners
/**
If the action is an extended action, all intermediated steps and the double step itself get recovered from the Episode object,
and send to the listeners (intermediate Steps gets send with the "intermediateStep" method). If the action is not an extended action,
the nextSend Method from the super class gets called.
*/
	virtual void sendNextStep(CAction *action);

 
	virtual bool isFinished(CStateCollection *, CStateCollection *) {return false;};

	virtual CAction *getNextHierarchyLevel(CStateCollection *stateCollection, CActionDataSet *actionDataSet = NULL);

	/// Add a state Modifier to the StateCollections
	virtual void addStateModifier(CStateModifier *modifier);
	/// remove a state Modifier from the StateCollections
	virtual void removeStateModifier(CStateModifier *modifier);

};

/// The class represents the main acting Object of the Learning System, the agent
/** The agent is the object which acts within its environment an sends every step to its 
SemiMDPListener, its the only "acting" object so it's the most important part of the toolbox. The Agent follows the Policy set by setController(CAgentController *). It saves the
currentState, then tells the model which action to execute and then saves the new state. Having done that the agent is able
to send the State-Action-State tuple to all its Listeners. \par
The agent's actionset can only maintain PrimitiveActions. The agent has an agent controller which can choose from the actions in the agent's actoinset. It is not allowed that an controller returns an action which isn't in the agent's action set.
ExtendedActions can only be added to the CHierarchicalSemiMarkovDecisionProcess class.
\par
Another important functionality of the agent are the StateModifiers which can be added to the agent. 
The stateModifier is than added to the stateCollections (currentState, lastState) of the agent. 
If you add a StateModifier to the agent, the modified state is calculated  by the state modifier after the modelstate has changed and added to the state collection.
The modified State, which can be a discrete, a feature or any other State calculated from the original model state is now 
available to all Listeners, and it only gets calculated once. So the Listeners have access to several different kind of states.
\par
For the execution of actions you have several possibilities, the agent can execute:
- a given action (doAction), 
- a single action from the controller (doControllerStep) 
- or one or more Episodes following the Policy from the Controller (doControllerEpisode). You can specify how much steps each episode should have at maximum.
The agent also loggs the current episode. You need the current Episode for the hierarchical SMDPs, they need 
an instance of the Episode to reconstruct the intermediate steps. This feature can be turned off by setLoggegEpisode(bool) for performance reasons if it isn't needed.
@see CSemiMDPListener
@see CHierarchicalSemiMarkovDecisionProcess
@see CSemiMarkovDecisionProcess
*/
class CAgent : public CSemiMarkovDecisionProcess, public CStateModifiersObject
{
protected:
	CStateCollectionImpl *currentState;
	CStateCollectionImpl *lastState;
	
    int maxEpisodes;
	int maxSteps;

	bool keyboardBreaks;

	CEnvironmentModel *model;

	bool bLogEpisode;

	int doRun(bool bContinue);

	CEpisode *currentEpisode;
public:

	CAgent(CEnvironmentModel *model);
	~CAgent();

/// Execute the action and send the State-Action-State tuple
	virtual void doAction(CAction *action);
/// Add a state Modifier to the StateCollections
	virtual void addStateModifier(CStateModifier *modifier);
/// remove a state Modifier from the StateCollections
	virtual void removeStateModifier(CStateModifier *modifier);



/// Executes maxEpisodes, if an epsiode reaches maxsteps, a new episode is startet automatically
/// Returns -1 if training has been paused by a keystroke.
/// Call doResume() to continue training.
	int doControllerEpisode(int maxEpisodes = 1, int maxSteps = 5000);
/// Set the Training Parameters, called by doControllerEpisode	
	void setParameters(int maxEpisodes, int maxSteps);
/// Resume the Training if it was paused (e.g. by a keystroke)	
/// Returns -1 if training has been paused by a keystroke.
/// Call doResume() to continue training.
	int doResume();

/// Tells all Listeners that a new Episode has occured and resets the model
	virtual void startNewEpisode();

/// Gets action from the controller and executes it
	/** 
	Be aware that you will get an assertation if the agent controller isn't set probably!
	*/
	void doControllerStep();

/// Sets wether the training can be paused by a keystroke. 
	void setKeyboardBreak(bool keyboardBreak);
    bool getKeyboardBreak();

/// add an primitiv action to the agent's actionlist. The agent can only choose from this actions.
	virtual void addAction(CPrimitiveAction *action);

/// Sets wether the currentEpisode should be logged or not.
	virtual void setLogEpisode(bool bLogEpisode);

/// Returns the currentEpisode Object (only valid if bLogEpisode = true).
	virtual CEpisode *getCurrentEpisode();

	virtual CStateCollection *getCurrentState();

	CEnvironmentModel *getEnvironmentModel();
};

class CHiearchicalAgent : public CAgent
{
	protected:
		CAgent *realAgent;
		CHierarchicalSemiMarkovDecisionProcess *hierarchicSMDP;
	public:
		CHiearchicalAgent(CAgent *agent, CHierarchicalSemiMarkovDecisionProcess *hierarchicSMDP);

		virtual ~CHiearchicalAgent();

		virtual void doAction(CAction *action);
		virtual void startNewEpisode();

};

#endif
