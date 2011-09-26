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

#ifndef C_ACTORCRITIC_H
#define C_ACTORCRITIC_H

#include "cerrorlistener.h"
#include "cagentlistener.h"
#include "cagentcontroller.h"


class CState;
class CStateCollection;
class CStateProperties;

class CAction;
class CActionData;
class CActionDataSet;

class CAbstractQFunction;
class CAbstractQETraces;

class CAbstractVFunction;
class CAbstractVETraces;

class CStochasticPolicy;

class CContinuousActionGradientPolicy;
class CGradientVETraces;
class CFeatureList;
class CContinuousActionData;


/// Interface for all Actors
/**The actors have to adopt their policies according to the critics they get. The class CActor provides an interface for sending a critic
to the actor (receiveError(CStateCollection *currentState, CAction *action). This is all the Actor classes have to implement. How to get the policy from
the actor is decontrolled to the Actor classes itself. The class also maintains a learning rate beta, which can be used by the subclasses.
*/
class CActor : public CErrorListener
{
protected:
	
public:
	CActor();
	
/// interface function for the actors
/** The actor gets a critic for a given state action pair. Then he has to adopt his policy according to the critic. */
	virtual void receiveError(double critic, CStateCollection *oldState, CAction *Action, CActionData *data = NULL) = 0;


	double getLearningRate();
	void setLearningRate(double learningRate);
};

/// Actor who creates his Policy on a Q Function
/**The CActorFromQFunction updates it's Q-Function on the particular state action pair
according to the critic he got for that state action pair. Since we are using a Q-Function the actor from Q-Function uses QETraces to boost learning. 
The policy from the actor is usually a Softmax Policy using the Q-Function, this Policy must be created by the user exclusivly. 
<p>
The Q-Function update for this actor is Q(s,a)_new = Q(s,a)_old + beta * td, where td is the value coming from the critic. 
@see CActorFromQFunctionAndPolicy. */
class CActorFromQFunction : public CActor, public CSemiMDPListener
{
protected:
	/// The Q Function of the actor
	CAbstractQFunction *qFunction;
	/// The Etraces used for the QFunction
	CAbstractQETraces *eTraces;

public:
/// Creates an Actor using the specified Q-Function to adopt his Policy. 
	CActorFromQFunction(CAbstractQFunction *qFunction);
	virtual ~CActorFromQFunction();

/// Updates the Q-Function
/**
The actor first updates the Etraces (i.e. mulitply all ETraces with gamma*lambda and then adds the state to the ETraces). 
Then the Q-Function is updated by the Etraces Object with the value beta * critic.
@see CQETraces
*/
	virtual void receiveError(double critic, CStateCollection *oldState, CAction *Action, CActionData *data = NULL);
/// Returns the used Q-Function
	CAbstractQFunction *getQFunction();
/// Returns the used ETraces
	CAbstractQETraces *getETraces();

	/// resets etraces object
	virtual void newEpisode();

};

/// Actor which uses a QFunction and his Policy for the update
/** The only difference to CActorFromQFunction is the update of the Q-Function. The update is
Q(s_t,a_t)_new = Q(s_t,a_t)_old + beta * td * (1 - pi_(s_t, a_t)), where pi(s_t, a_t) is the softmax-policy from the actor. This method is recommended by
Sutton and Barto.
*/
class CActorFromQFunctionAndPolicy : public CActorFromQFunction
{
protected:
	CStochasticPolicy *policy;
	double *actionValues;

public:
/// Creates the actor object, the policy has to choose the actions using the specified Q-Function.
	CActorFromQFunctionAndPolicy(CAbstractQFunction *qFunction, CStochasticPolicy *policy);
	virtual ~CActorFromQFunctionAndPolicy();

/// Updates the Q-Function
/** Does the following update: Q(s_t,a_t)_new = Q(s_t,a_t)_old + beta * td * (1 - pi(s_t, a_t))
*/
	virtual void receiveError(double critic, CStateCollection *state, CAction *Action, CActionData *data = NULL);

	CStochasticPolicy *getPolicy();

	
};

/// Actor class which can only decide beween 2 different action, depending on the action value of the current state
/** 
This is the implementation of the simple Actor-Critic Algorithm used by Barto, Sutton, and Anderson in their cart pole example. The actor can only decide between 2 actions. Which action is taken depends on the action value of the current state. If this value is negative, the first action is more likely to be choosen and vice versa. The probabilty of choosing the first action is caculated the following way : 1.0 / (1.0 + exp(actionvalue(s))).
The action weight value is represented by an V-Function, for updating the V-Function an etrace object is used. The current state is added to the etrace with a positive factor if the second action was choosed, otherwise with a negative factor. When a new episode begins, the etraces are resetted.
This kind of algorithm usually need a very high learning rate, for this class 1000.0 is the standard value for the "ActorLearningRate" Parameter.
<p>
This class directly implements the CAgentController interface, so it can be used as controller.
*/
class CActorFromActionValue : public CAgentController, public CActor, public CSemiMDPListener
{
protected:
	CAbstractVFunction *vFunction;
	CAbstractVETraces *eTraces;

public:
	CActorFromActionValue(CAbstractVFunction *vFunction, CAction *action1, CAction *action2);
	~CActorFromActionValue();

	/// Adopt the action values according to the critic
	virtual void receiveError(double critic, CStateCollection *oldState, CAction *Action, CActionData *data = NULL);

	virtual CAction *getNextAction(CStateCollection *state, CActionDataSet *data = NULL);
		/// resets etraces object
	virtual void newEpisode();
};

class CActorFromContinuousActionGradientPolicy : public CActor, public CSemiMDPListener
{
protected:
	CContinuousActionGradientPolicy *gradientPolicy;
	CGradientVETraces *gradientETraces;
	CFeatureList *gradientFeatureList;

	CContinuousActionData *policyDifference;
public:
	CActorFromContinuousActionGradientPolicy(CContinuousActionGradientPolicy *gradientPolicy);
	virtual ~CActorFromContinuousActionGradientPolicy();

	virtual void receiveError(double critic, CStateCollection *oldState, CAction *Action, CActionData *data = NULL);
	virtual void newEpisode();
};


class CActorForMultipleAgents : public CActor, public CAgentController
{
	protected:
		std::list<CActor *> *actors;
		std::list<CAgentController *> *actionSets;
		unsigned int numActions ;
			
	public:
		
		CActorForMultipleAgents(CActionSet *actions);

		virtual ~CActorForMultipleAgents();


		void addActor(CActor *actor, CAgentController *policy);

		virtual void receiveError(double critic, CStateCollection *state, CAction *action,  CActionData *data);

		virtual CAction* getNextAction(CStateCollection *state, CActionDataSet *dataset);
};


#endif
