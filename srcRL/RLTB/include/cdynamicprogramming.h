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

#ifndef C_DYNAMICPROGRAMMING__H
#define C_DYNAMICPROGRAMMING__H


#include "cparameters.h"


#include <map>


class CTransition;
class CAbstractFeatureStochasticModel; 
class CFeatureRewardFunction;
class CAbstractVFunction;
class CState;
class CAction;
	

class CFeatureQFunction;
class CFeatureVFunction;
class CQFunctionFromStochasticModel;
class CActionSet;

class CFeatureList;
class CStochasticPolicy;

/// Collection of static functions for dynamic Programming
/** Provides Functions for Calculating the Action Value, the Bellman Value and the Bellman Error given a theoretical model, a V-Function and a reward function for a given state.
*/
class CDynamicProgramming 
{
public:
	/// Calculates the Action Value of the given state action pair
	/**
	The action value of a action a in state s is defined through Q(s,a)=sum_{s'}P(s'|s,a)*(R(s,a,s')+gamma* V(s')). 
	The Propabilities P(s'|s,a) come from the forward transitions of the model for the given state-action pair. The forward Transitiions are iterated, the
	expected total discount reward is calculated P(s'|s,a)*(R(s,a,s')+gamma* V_(s')) and the expectation of this value, which is the action value, is calculated.
	For the semi MDP case the formulae is a bit more complex, Q(s,a)=sum_{s',N}P(s',N|s,a)*(R(s,a,s')+gamma^N* V(s')). This formulae is used if the specified action
	is a multistep action, the Transition objects are then CSemiMDPTransition objects, which also stores the probapilities of the durations.
	R(s, a, s') comes obviously from the reward Function, which has to be a feature Reward Function, because the Reward for the Feature(Díscrete State)-Transitions are needed.
	The given state has to be a discrete state, and the action has to be member of the model.
	@see CTransition
	@see CSemiMDPTransition
	*/
	static double getActionValue(CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunc, CAbstractVFunction *vFunction, CState *discState, CAction *action, double gamma);
	/// Calculates the BellmanValue, which is the best value achievable in the current State, given a Value Function and a Reward Function
	/** Since the BellmanValue is the best Value achievable, its the best action Value. So the function calculates V^*(s)=max_a Q(s,a), the action Values come from
	the function getActionValue. The given state has to be a discrete state*/
	static double getBellmanValue(CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunc, CAbstractVFunction *vFunction, CState *discState, double gamma);
	/// Calculates the Bellman Error of the Value Function in the given state
	/** The Bellman error is just the Bellman Value minus the Value of the V-Function for the current state.
	*/
	static double getBellmanError(CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunc, CAbstractVFunction *vFunction, CState *discState, double gamma);
};

/// The Value Iteration Algorithm
/**
Value Iteration calculates the Value Function of a arbitrary policy for a given learning problem, it expects a given stochastic model of the learning problem, so if you need to learn the model as well, use the prioritized sweeping algorithm.
The Value iteration classes of the toolbox provides both, V-Function learning and Q-Function learning.
Value iteration uses the update rule V_{k+1}=sum_a pi(s,a)*sum_{s'}P(s'|s,a)*(R(s,a,s')+gamma* V_(s')) (where pi is a stochastic policy) for value function learning and Q(s,a)= sum_{s'}P(s'|s,a)*(R(s,a,s')+gamma* V_(s')), where $V_k(s') = sum_a Q(s',a)*pi(s',a)$ for the Q-Value learning case. If you repeat that step arbitrary often, the update rule converges to the value function of the policy.
Usually a greedy policy is used for learning, since you want the optimal value function, but you can also choose to evaluate the value function of some other, maybe self-coded policy (as long as it implements the interface CStochasticPolicy).
Dynamic Programming approaches are usually a safe tool to gather the optimal value function, but it is also a very CPU-intensive task, so it is very important which state is updated because in the most states the update is very small or even zero. 
So the class CValueIteration also maintains a priority list of the states, indicating which state has to be updated first. If a state is updated according to the given rules, the error of the former value is calculated and than every state in the backward list of the updated state from the stochastic model (so every state which leads to the updated state), gets his priority added by the value error * prop, where prop is the probability of that (backward) transition. This concept comes from prioritized sweeping.
Due to this concept the states which are likely to change their Values considerably gets updated first. The class provides functions for updating the states in the priority list k times (if the list is empty a random state is chosen), update the states until the list is empty, or update a single given state.
To give the algorithm a little hint where to start you can also update all features in the backward transitions of a specific state. 
For the priority List the algorithm uses a sorted feature list.
\par
You can choose if you want to learn a Value-Function or directly a Q-Function by providing a Q-Function or a Value Function to the constructor. Learning a QFunction can have the advantage that this Q-Function can be
used by other learning algorithms too. If you use a V-Function you have to get a QFunction for the policies from the VFunction, this is done by CQFunctionFromStochasticModel, which takes the stochastic model and a VFunction and calculates the Q-Values if they are requested.
The update process works as follows:
<ul>
<li> Learning with the V-Function: The new Value of the state is calculated by V_{k+1}=sum_a pi(s,a)*sum_{s'}P(s'|s,a)*(R(s,a,s')+gamma* V_(s')), then the error is calculated and used for priority updates. </li>
<li> Learning with the Q-Function: The Value of the state is calculated by V_k(s') = sum_a Q(s',a)*pi(s',a), this is done by the class CVFunctionFromQFunction. Then each action-value is updated by 
Q(s,a)= sum_{s'}P(s'|s,a)*(R(s,a,s')+gamma* V_(s')), after update the new V-Value is calculated to get the error needed for the priorities. </li>
</ul>
If you use Q-Function learning, a CVFunctionFromQFunction is used for the calculation of the Value, if you use V-Function learning a CFeatureQFunctionFromVFunction is used for the calculation of the action Values.
<p>
You can also specify a policy which Value or QFunction you want to learn, so you can do policy evaluation if the policy is fixed. The standard policy is the greedy policy, so you calculate the optimal Value function.
*/

class CValueIteration : virtual public CParameterObject
{
protected:
/// The used V-Function
	CAbstractVFunction *vFunction;
/// V-Function used for the new Value calculation when using V-Function Learning
	CAbstractVFunction *vFunctionFromQFunction;
/// The used Q-Function
	CFeatureQFunction *qFunction;
/// Q-Function for the Action Value calculation when using V-Learning
	CQFunctionFromStochasticModel *qFunctionFromVFunction;
/// the model
	CAbstractFeatureStochasticModel *model;
/// reward function of the learning Problem
	CFeatureRewardFunction *rewardModel;
/// The actions used by the value iteration
	CActionSet *actions;

/// use V or Q Function?
	bool learnVFunction;
/// Temporary state object
	CState *discState;

/// Sorted list of the priorities
	CFeatureList *priorityList;

/// The stochastic Policy which is used.
	CStochasticPolicy *stochPolicy;

/// returns the priority of a specific Transition given the bellman error
/** The standard priority calculation is trans->getProbapility() * bellE, but this can be changed by possible subclasses
*/
	virtual double getPriority(CTransition *trans, double bellE);
	void init(CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel);

  
public:
/// Creates the Value Iteration algorithm with Q-Function learning and a greedy policy
	CValueIteration(CFeatureQFunction *qFunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel);
/// Creates the Value Iteration algorithm with Q-Function learning and given policy for policy evaluation
	CValueIteration(CFeatureQFunction *qFunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel, CStochasticPolicy *stochPolicy);
/// Creates the Value Iteration algorithm with Q-Function learning and a greedy policy
	CValueIteration(CFeatureVFunction *vFunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel);
/// Creates the Value Iteration algorithm with Q-Function learning and given policy for policy evaluation
	CValueIteration(CFeatureVFunction *vFunction, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardModel, CStochasticPolicy *stochPolicy);
	virtual ~CValueIteration();

/// Updates the given feature
/** Clears the feature from the prioritylist and then makes either a Q-Function or a V-Function update. 
The update process works as follows:
<ul>
<li> Learning with the V-Function: The new Value of the state is calculated by V_{k+1}=sum_a pi(s,a)*sum_{s'}P(s'|s,a)*(R(s,a,s')+gamma* V_(s')), then the error is calculated and used for priority updates. </li>
<li> Learning with the Q-Function: The Value of the state is calculated by V_k(s') = sum_a Q(s',a)*pi(s',a), this is done by the class CVFunctionFromQFunction. Then each action-value is updated by
Q(s,a)= sum_{s'}P(s'|s,a)*(R(s,a,s')+gamma* V_(s')), after update the new V-Value is calculated to get the error needed for the priorities. <li>
</ul>
After that alls backwards states are fetched from the model and added to the priority list with the priority getPriority(transition, bellError), which is in
standard transition->getPropability() * bellError.*/
	virtual void updateFeature(int feature);

/// Updates the first feature from the list
	void updateFirstFeature();

/// Adds the given priority to the given feature
	void addPriority(int feature, double priority);
/// Add all Priorities of the featuers in the feature list
	void addPriorities(CFeatureList *featList);

	CAbstractFeatureStochasticModel *getTheoreticalModel();
	CAbstractVFunction *getVFunction();
	CFeatureQFunction *getQFunction();
	CStochasticPolicy *getStochasticPolicy();
	
	int getMaxListSize();
	void setMaxListSize(int maxListSize);

/// updates the frist k states in the priority list
/** If the list is empty, a random state is chosen*/
	void doUpdateSteps(int k);
/// Updates the states from the priority list until it is empty.
	void doUpdateStepsUntilEmptyList(int k);

/// Updates all backward states of the given state
/** Used to give the algorithm a hint where to start, since due to the updates, all backward states of the backward states 
gets added to the prioritylist (as long as they made a Bellman error.
*/
	void doUpdateBackwardStates(int state);
};


#endif

