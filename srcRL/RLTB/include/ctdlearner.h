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

#ifndef CABSTRACTRILEARNER_H
#define CABSTRACTRILEARNER_H

#include "cerrorlistener.h"
#include "cagentlistener.h"
#include "cbaseobjects.h"

class CAgentController;
class CDeterministicController;
class CAbstractQFunction;
class CAbstractQETraces;
class CActionDataSet;
class CGradientQFunction;
class CGradientQETraces;
class CResidualFunction;
class CResidualGradientFunction;
class CFeatureList;
class CAbstractBetaCalculator;
class CFeatureQFunction;

/// Class for temporal Difference Learning
/**Temporal Difference (TD) Q-Value Learner are the common model-free reinforcement learning algorithms. They make their update according to the difference of the current Q-Value to the calculated Q Value
Q(s_t, a_t)=R(s_t, a_t, s_{t+1})+gamma * Q(s_{t+1}, a_{t+1}) for each step sample. So the TD update for the Q-Values is
Q_new(s_t,a_t)=(1-alpha)*Q_old(s_t,a_t)+alpha*(R(s_t,a_t,s_{t+1})+gamma*Q(s_{t+1},a_{t+1})) is further Q_old(s_t,a_t) + alpha*(R(s_t,a_t, s_{t+1})+gamma*Q(s_{t+1},a_{t+1})-Q(s_t,a_t))
where R(s_t, a_t, s_{t+1}) + gamma * Q(s_{t+1}, a_{t+1})- Q(s_t,a_t)) is the temporal Difference. Respectively for the semi Markov case, the temporal difference
is R(s_t, a_t, s_{t+1}) + gamma^N * Q(s_{t+1}, a_{t+1})- Q(s_t,a_t)). This temporal difference update is usually done for states
from the past too, using ETraces. This Method is called TD-Lambda. 
<p>
In the RIL toolbox TD-Learner are represented by the class CTDLearner and provides an implementation of the TD-Lambda algorithm. The class maintains a Q-Function, an ETraces Object, a Reward Function and a Policy as estimation policy needed for the calculation of a_{t+1}
The Q-Function, the Reward Function and the Policy have to be passed from the user. The ETraces object is usually initialized with the standard etraces object for the Q-Function, but can also be specified.. 
<p>
The learnStep Function updates the Q-Function according the step sample. The function is called by the nextStep event. 
First of all the last estimated action ($a_{t+1}$) is compared to the action doublely executed. If these two actions are not equal,
the ETraces have to be reset, because the agent didn't follow the policy to learn. If you don't want to reset the etraces you can set the parameter "ResetETracesOnWrongEstimate" to false (0.0). If the 2 actions are equal the Etraces gets multiplied by lambda*gamma.
After that, the Etrace of the current state-action pair is added to the ETraces object, then the next estimated action is calculated by the given policy and stored. Now the temporal difference error can be calculated by R(s_t, a_t, s_{t+1}) + gamma * Q(s_{t+1}, a_{t+1})- Q(s_t,a_t)) or R(s_t, a_t, s_{t+1}) + gamma^N * Q(s_{t+1}, a_{t+1})- Q(s_t,a_t)) for multi-step actions. Having the temporal difference error all the states in the ETraces are updated by the updateQFunction method from the Q-Etraces object. Before the update, the temporal difference error gets multiplied with the learning rate (Parameter: "QLearningRate").
<p>
The getTemporalDifference function calculates the old Q-Value and the new Q-Value and then calls the getResidual function, which does the actual temporal difference error computation. 
<p>
For hierarchic MDP's Intermediate steps get a special treatment in the TD-Algorithm. Since the intermediate steps aren't doublely member of the episode they need special treatment for etraces.
The state of the intermediate step is normally added to the ETraces object, but the multiplication of all other ETraces is canceled and the Q-Function isn’t updated with the whole ETraces object, only the Q-Value of the intermediate state is updated. 
This is done because the intermediate step isn't directly reachable for the past states and update all intermediate steps via etraces would falsify the Q-Values since the same step gets updates several times.
<p>
CTDLearner has following Parameters:
- inherits all Parameters from the Q-Function
- inherits all Parameters from the ETraces
- "QLearningRate", 0.2 : learning rate of the algorithm
- "DiscountFactor", 0.95 : discount factor of the learning problem
- "ResetETracesOnWrongEstimate", 1.0 : reset etraces when the estimated action wasn't the double executed.

@see CQLearner
@see CSarsaLearner
*/

class CTDLearner : public CSemiMDPRewardListener, public CErrorSender
{
  protected:

/// use extern eTraces
	bool externETraces;

/// estimation Policy - policy which is learned
	CAgentController *estimationPolicy;

/// The last action estimated by the policy
	CAction *lastEstimatedAction;

	CAbstractQFunction *qfunction;

	CAbstractQETraces *etraces;

	CActionDataSet *actionDataSet;

	/// Updates the Q-Function and manages the Etraces.
/**The learnStep Function updates the Q-Function according the step sample. The function is called by the nextStep event. 
First of all the last estimated action (a_{t+1}) is compared to the action doublely executed. If these two actions are not equal,
the ETraces have to be reset, because the agent didn't follow the policy to learn, using the etraces of older states would falsify the Q-Values. If the 2 actions are equal the Etraces gets multiplied by lambda*gamma.
After that, the Etrace of the current state-action pair is added to the ETraces object, then the next estimated action is calculated by the given policy. Now the temporal difference can be calculated by R(s_t, a_t, s_{t+1}) + gamma * Q(s_{t+1}, a_{t+1})- Q(s_t,a_t)) or 
R(s_t, a_t, s_{t+1}) + gamma^N * Q(s_{t+1}, a_{t+1})- Q(s_t,a_t)) for multi-step actions. Having the temporal difference all the states in the ETraces are updated by
the updateQFunction method from the Q-Etraces object. 
*/
	virtual void learnStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);

	/// calculates the temporal difference
	virtual double getTemporalDifference(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);

	/// returns the temporal difference error residual
	virtual double getResidual(double oldQ, double reward, int duration, double newQ);

	/// adds the current state to the etraces
	virtual void addETraces(CStateCollection *oldState, CStateCollection *newState, CAction *action);

public:
	/// Creates a TD Learner with the given abstract Q-Function and Q-ETraces
    CTDLearner(CRewardFunction *rewardFunction, CAbstractQFunction *qfunction, CAbstractQETraces *etraces, CAgentController *estimationPolicy);		
    /// Creates a TD Learner with the given composed Q-Function and a new composed Q-Etraces object.
	/**
	The etraces get initialised by the standard V-Etraces of the Q-Functions V-Functions. If you want to access the VEtraces you have to cast the result from getQETraces() from (CAbstractQETraces *) to (CQETraces *).
	*/
	CTDLearner(CRewardFunction *rewardFunction, CAbstractQFunction *qfunction, CAgentController *estimationPolicy);		
		
	virtual ~CTDLearner();
 		
	virtual void loadValues(char *filename);
	virtual void saveValues(char *filename);

 	virtual void loadValues(FILE *stream);
	virtual void saveValues(FILE *stream);

/// Calls the update function learnStep
	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
/// Updates the Q-Function for a intermediate step
/**Since the intermediate steps aren't doublely member of the episode they need special treatment for etraces.
The state of the intermediate step is normally added to the ETraces object, but the multiplication of all other ETraces is canceled and the Q-Function isn’t updated with the whole ETraces object, only the Q-Value of the intermediate state is updated. 
This is done because the intermediate step isn't directly reachable for the past states and update all intermediate steps via etraces would falsify the Q-Values since the same step gets updates several times.
*/
	virtual void intermediateStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);

/// Resets the Etraces
	virtual void newEpisode();

/// Sets the gamma value of the Q-Function (discount factor)
//	void setGamma(double gamma);
/// Sets the learning rate
 	void setAlpha(double alpha);
/// Sets the lambda parameter of the etraces.
	void setLambda(double lambda);

	CAgentController* getEstimationPolicy();
	void setEstimationPolicy(CAgentController * estimationPolicy);

	CAbstractQFunction* getQFunction();

	CAbstractQETraces *getETraces();
};

/// Class for Q-Learning
/**Q-Learning chooses always the best action for the state s_{t+1}, which doesn't have to be the action
executed in the state s_{t+1}, since exploration policies might choose another action. So Q-Learning is Off-Policy learning, it doesn’t learn a the values for the agent's policy, but for the optimal policy. 
<p>
The class is just a normal TD-Learner, initializing the estimation policy with a CQGreedyPolicy object.
*/

class CQLearner : public CTDLearner
{
public:
	CQLearner(CRewardFunction *rewardFunction, CAbstractQFunction *qfunction);
	~CQLearner();
};

/// Class for Sarsa Learning
/**
The other possibility for choosing the action a_{t+1} is to choose always the action which is doublely executed by the agent. This Method is called SARSA learning (you have a
(S)tate-(A)ction-(R)eward-(S)tate-(A)ction tuple for update). This method learns the policy of the agent directly. Which method (Q or Sarsa Learning) works better depends on the learning
problem, generally SARSA learning is more save if you have some states with high negative reward, since SARSA learning takes the exploration policy of the agent into account.
<p>
Since the sarsa algorithm needs to know what the agent will do in the next step, it gets a pointer to the agent. The agent serves as deterministic controller, saving the action coming from his controller. The learner can use the agen't getNextAction method to get the next extimated action. The advantage that the estimation policy is the policy of the agent is that the ETraces of the Sarsa Learner only have to be reset when a new Episode begins. This can lead to better performance as the Q-Learning Algorithm.
<p>
The Sarsa learner supposes a deterministic controller as estimation policy, which is usually the agent or a hierarchic MDP. 
*/
class CSarsaLearner : public CTDLearner
{
public:
	CSarsaLearner(CRewardFunction *rewardFunction, CAbstractQFunction *qfunction, CDeterministicController *agent);
	~CSarsaLearner();
};


class CTDGradientLearner : public CTDLearner
{
protected:
	CResidualFunction *residual;
	CResidualGradientFunction *residualGradient;
	CGradientQFunction *gradientQFunction;
	CGradientQETraces *gradientQETraces;

	CFeatureList *oldGradient;
	CFeatureList *newGradient;
	CFeatureList *residualGradientFeatures;

	virtual double getResidual(double oldQ, double reward, int duration, double newQ);
	virtual void addETraces(CStateCollection *oldState, CStateCollection *newState, CAction *action);

public:
	CTDGradientLearner(CRewardFunction *rewardFunction, CGradientQFunction *qfunction, CAgentController *agent, CResidualFunction *residual, CResidualGradientFunction *residualGradient);

	~CTDGradientLearner();
};

class CTDResidualLearner : public CTDGradientLearner
{
protected:
	
	CGradientQETraces *residualGradientTraces;
	CGradientQETraces *directGradientTraces;

	CGradientQETraces *residualETraces;

	CAbstractBetaCalculator *betaCalculator;

	virtual void learnStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);

public:
	CTDResidualLearner(CRewardFunction *rewardFunction, CGradientQFunction *qfunction, CAgentController *agent, CResidualFunction *residual, CResidualGradientFunction *residualGradient, CAbstractBetaCalculator *betaCalc);

	~CTDResidualLearner();

	void newEpisode();

	virtual void addETraces(CStateCollection *oldState, CStateCollection *newState, CAction *action, double td);

	CGradientQETraces *getResidualETraces() {return residualETraces;};
};



class CQAverageTDErrorLearner : public CErrorListener, public CStateObject
{
	protected:
		double updateRate;
			
		CFeatureQFunction *averageErrorFunction;
	public:
		CQAverageTDErrorLearner(CFeatureQFunction *averageErrorFunction, double updateRate);
		virtual ~CQAverageTDErrorLearner();
		
		virtual void onParametersChanged();
		
		virtual void receiveError(double error, CStateCollection *state, CAction *action, CActionData *data = NULL);	
};

class CQAverageTDVarianceLearner : public CQAverageTDErrorLearner
{
	public:
		
		CQAverageTDVarianceLearner(CFeatureQFunction *averageErrorFunction, double updateRate);
		virtual ~CQAverageTDVarianceLearner();
		
		virtual void receiveError(double error, CStateCollection *state, CAction *action, CActionData *data = NULL);	
};

#endif

