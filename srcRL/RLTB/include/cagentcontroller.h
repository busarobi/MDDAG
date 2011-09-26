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

#ifndef CAGENTCONTROLLER_H
#define CAGENTCONTROLLER_H

#include "cbaseobjects.h"
#include "cagentlistener.h"
#include "cparameters.h"


#include <list>

class CStateCollection;
class CActionStatistics;

/// Class for navigating the agent
/**The agent has an own agent controller, which tells the agent what to do. Agent controllers return an action given a specific state collection.
This is done by the interface CAction* getNextAction(CStateCollection *state), thsi function must be implemented by all subclasses.
Agent controllers have an own action set they can choose from, but the agent controllers must return one action which is member of the allocated semi MDP's action set.
So the controller navigating the agent can only choose from primitive actions.
*/

class CAgentController : public CActionObject, virtual public CParameterObject
{
public:
/// Constructor for the controller, sets the agent controllers action set.
	CAgentController(CActionSet *actions);
	virtual ~CAgentController();

/// Virtual function for returning the action for the specified state, must be implemented by all subclasses.
	virtual CAction* getNextAction(CStateCollection *state, CActionDataSet *data = NULL) = 0;
	
};

/// Agent controller which returns additionally an statistic object for the action
/** The statistic object gives information about how good the controller things the choosed action is.
This is used by CMultiController, to choose the best controller in the specified state. Almost all implemented
policies are statistic controllers, returning the goodness of the q-value in comparison to the other actions.
@see CActionStatistics
*/
class CAgentStatisticController : public CAgentController
{
public:
	CAgentStatisticController(CActionSet *actions);

	virtual CAction* getNextAction(CStateCollection *state, CActionDataSet *data = NULL);
	virtual CAction* getNextAction(CStateCollection *, CActionDataSet * = NULL, CActionStatistics * = NULL) {return NULL;};
};

/// Controller class makes a given controller deterministic.
/** Many other controllers are stochastic controllers, which means that they don't return the same
action for the same state (even in the same step). So if getNextAction is called more than once during a step,
the resulting action don't have to be the same. For some algorithm a deterministic behavior (at least for one step) is needed, so they can ask for the action which has been chosen in the current step (e.g. CSarsaLearner needs the exact policy of the agent). The CDeterministicController  isn't deterministic for states (same state -> same action) but for steps (same step -> same action), it stores the action computed by the given controller each time the getNextAction is called and a newStep begins. It always returns the stored action until a new step begins again.
<p>
The class CDeterministicController makes the stochastic controller returning the same action while being in the same 
step. Therefore it stores the calculated action in the nextAction field, returning that action when further requests for the action occure.
When the step is finished (nextStep event occurs) the nextAction is cleared and with the first request newly calculated.
To gather the nextStep and newEpisode events the controller HAS TO BE ADDED to the agent's listeners. The deterministic controller
can handle statistic controllers and normal controllers. When using a statistic controller the statistic object is stored
in the statistics field and can be obtained by getLastActionStatistics().
For actions with changeable action data, the action data is stored in actionDataSet, so the changeable action data is calculated only once too.
@see CSemiMarkovDecisionProcess
*/

class CDeterministicController : public CAgentController, public CSemiMDPListener
{
protected:
	/// field for storing the calculated action
	CAction *nextAction;
	/// the stochastic controller 
	/**
	only one of controller and statisticController is used.
	*/
	CAgentController *controller;
	/// the stochastic statistic controller
	/**
	only one of controller and statisticController is used.
	*/
	CAgentStatisticController *statisticController;
	/// last statistics returned by the statistic controller, not used when using a normal controller
	CActionStatistics *statistics;
	bool useStatisticController;

	/// action Data set of all actions, the controller uses this data set to store the changeable actoin data until the nextstep.
	CActionDataSet *actionDataSet;

	void initStatistics();

public:
	///creates a new deterministic controller, 
	CDeterministicController(CActionSet *actions);
	///creates a new deterministic controller, with the given controller to make deterministic
	CDeterministicController(CAgentController *controller);
	///creates a new deterministic controller, with the given statistc controller to make deterministic
	CDeterministicController(CAgentStatisticController *controller);
	virtual ~CDeterministicController();

	/// returns the action stored in nextAction
	/** if the nextAction was cleared, a new action is calculated and stored in the nextAction field
	*/
	virtual CAction* getNextAction(CStateCollection *state, CActionDataSet *data = NULL);

	/// Clears the nextAction, forcing a new calculation of the action
	virtual void nextStep(CStateCollection *state1, CAction *action, CStateCollection *state2);
	/// Clears the nextAction, forcing a new calculation of the action
	virtual void newEpisode();
    
	/// returns the last statistics got from the statistic controller.
	/** object is empty or uninitialized when no statistic controller is used. */
	CActionStatistics *getLastActionStatistics();
    bool isUsingStatisticController();
	
	/// set the controller, no statistic controller is used
	void setController(CAgentController *controller);
	/// set the controller
	/** so a statistic controller is used and the statistics of the controller can be obtained by getLastStatistics*/
	void setController(CAgentStatisticController *controller);

	CAgentController *getController() {return controller;};

	void setNextAction(CAction *action, CActionData *data = NULL);
};


#endif
