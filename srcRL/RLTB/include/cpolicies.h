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

#ifndef CTDLEARNERPOLICIES_H
#define CTDLEARNERPOLICIES_H

#include "cagentcontroller.h"
#include "cparameters.h"

class CAbstractFeatureStochasticEstimatedModel;
class CTransitionFunction;
class CAbstractQFunction;
class CActionSet;
class CFeatureList;
class CActionStatistics;
class CAbstractVFunction;
class CQFunctionFromTransitionFunction;
class CStateCollectionImpl;

#include "newmat/newmat.h"

/// Greedy Policy based on a Q-Function
/** 
This policy always takes the greedy action (action with the highest Q-Value). The policy can't be used as stochastic policy, if a stochastic greedy policy is needed take CQStochasticPolicy with a greedy distribution.
*/
class CQGreedyPolicy : public CAgentController
{
protected:
	CAbstractQFunction *qFunction;
	CActionSet *availableActions;
public:
	CQGreedyPolicy(CActionSet *actions, CAbstractQFunction *qFunction);
	~CQGreedyPolicy();

	/// Always returns the greedy action
	virtual CAction *getNextAction(CStateCollection *state, CActionDataSet *data = NULL);

};

/// Action Distribution classes define the distributions of stochastic Policies
/** 
Action Distribution calculate the distribution for sampling an action, which is done by the class CStochasticPolicy. The distribution calculation usually depends on some kind of Q-Value of the actions. This is done in the function getDistribution. The function gets as input the current state, all available actions, and the Q-Values (actually it can be any kind of value, rating an action) of the actions as a double array. Usually only this Q-Values are used for the distribution (the state is only used for special exploration policies). The function has to overwrite the Q-Values double array with the distribution values.
Additionally  some algorithm needs a differntiable distribution. Therefore the interface provides the function isDifferentiable (since not all distributions are differentiable) and the function getGradientFactors. The function calculates the gradient dP(usedaction|actionFactors)/ (d_actionfactors). The actionfactors are again some kind of rating for the actions. The result has to be written in the output vector gradientfactors. This vector has always the same size as the actionfactors array (so the number of actions). Only the SoftMax Distribution supports calculating this gradient.  
*/
class CActionDistribution : virtual public CParameterObject
{
public:
	/// Returns the distribution of the actions that is sampled by an stochastic policy
/** 
The function gets as input the current state, all available actions, and the Q-Values (actually it can be any kind of value, rating an action) of the actions as a double array. Usually only this Q-Values are used for the distribution (the state is only used for special exploration policies). The function has to overwrite the Q-Values in double array with the distribution values.
*/
	virtual void getDistribution(CStateCollection *state, CActionSet *availableActions, double *actionFactors) = 0;
	virtual bool isDifferentiable() {return false;};

/// Calculates the derivation of the probability of choosing the specified action.
/**	 The function calculates the gradient dP(usedaction|actionFactors)/ (d_actionfactors). The actionfactors are again some kind of rating for the actions. The result has to be written in the output vector gradientfactors. This vector has always the same size as the actionfactors array (so the number of actions). Only the SoftMax Distribution supports calculating this gradient.*/
	virtual void getGradientFactors(CStateCollection *state, CAction *usedAction, CActionSet *actions, double *actionFactors, ColumnVector *gradientFactors);
};

///Soft Max Distribution for Stochastic Policies. 
/**
This class implements the well known softmax distribution (sometimes calles Gibs distribution). The Softmax Distribution is differentiable and therefore can be used for policy gradient algorithms. The Distribution depends on the parameter "SoftMaxBeta" which specifies you the "greediness" of your distribution.
<p>
The class CSoftMaxDistribution has the following Parameters: 
- "SoftMaxBeta" : Greediness of the distribution 
*/

class CSoftMaxDistribution : public CActionDistribution
{
protected:
public:

	CSoftMaxDistribution(double beta);

	virtual void getDistribution(CStateCollection *state, CActionSet *availableActions, double *values);

	virtual bool isDifferentiable() {return true;};

	virtual void getGradientFactors(CStateCollection *state, CAction *usedAction, CActionSet *actions, double *actionFactors, ColumnVector *gradientFactors);

};

class CAbsoluteSoftMaxDistribution : public CActionDistribution
{
protected:
public:

	CAbsoluteSoftMaxDistribution(double maxAbsValue);

	virtual void getDistribution(CStateCollection *state, CActionSet *availableActions, double *values);

	virtual bool isDifferentiable() {return false;};

	//virtual void getGradientFactors(CStateCollection *state, CAction *usedAction, CActionSet *actions, double *actionFactors, ColumnVector *gradientFactors);
};

///Class for a greedy action distribution. 
/** 
This class implements a greedy action distribution, so the probability for the best rated action is always 1, and for the rest 0. If there are more than one greedy action, always the first action will be taken. Its understood that this distribution is not differentiable. 
*/

class CGreedyDistribution : public CActionDistribution
{
public:
	virtual void getDistribution(CStateCollection *state, CActionSet *availableActions, double *values);
};

/// Class for the epsilon greedy action distribution. 
/**This class implements the epsilon greedy action distribution. Epsilon greedy policies take the greedy (best rated) action with probability (1 - epsilon) and a random action with probability epsilon. If there are more than one greedy action, always the first action will be taken. To set epsilon please use the parameter "EpsilonGreedy" or the constructor of the class. Its understood that this distribution is not differentiable.
<p>
The class CEpsilonGreedyDistribution has following Parameters:
- "EpsilonGreedy" : epsilon  
*/
class CEpsilonGreedyDistribution : public CActionDistribution
{
protected:
public:
//	double epsilon;

	CEpsilonGreedyDistribution(double epsilon);
	virtual void getDistribution(CStateCollection *state, CActionSet *availableActions, double *values);
};


/// Class for modeling a stochastic policy. 
/**
Many algorithm need more than just a specific action for a specific state, especially when the policy is a stochastic policy very often the distribution for choosing an action is needed. This is modeled by CStochasticPolicy. The Policy choses an action according to a given propability distribution, you can specify this distribution in the constructor with the CActionDistribution object. In the getNextAction Method an action is chosen according the distribution returned by getActionProbabilities. The getActionProbabilities method has to call the getDistribution method from the CActionDistribution object with the action rating as input. How this action rating is calculated has to be implemented by the subclasses, usually the values comes from a Q-Function (see CQStochasticPolicy). Some algorithms like the policy gradient algorithm need a differentiable action distribution. CStochasticPolicy also provides an interface for differentiate your distribution with respect to the policy weights (weights of the Q-Function). 

The gradient calculation of the policy is already implemented. You have the possibility to calculate dP(action| state)/ dweights or the logarithmic gradient which is the same as dP(action| state)/ dweights * 1 / P(action | state). Calculating the gradient of the action ratings (e.g. dQ(a,s)/dw for QFunctions) has to be implemented in the function getActionGradient if the stochastic policy is supposed to be differentiable. Differentiable policies also have to overwrite the function isDifferentiable, which always returns false for the base class. Wether the policy is differentiable or not depends on the kind of action ratings and on the distribution. Both of them have to be differentiable. 
The class als provides the possibility to get a statistics object for the action which was chosed. This is done by the virtual function getActionStatistics, which is called by the getNextAction Function if an statistics object is requestet. 
*/


class CStochasticPolicy: public CAgentStatisticController
{
protected:
/// array to store the current action propabilites
	double *actionValues;
	CActionDistribution *distribution;

	ColumnVector *gradientFactors;

	CFeatureList *actionGradientFeatures;
	
	CActionSet *availableActions;

/// virtual function for gettin the action statistic for the chosen action
/**The class als provides the possibility to get a statistics object for the action which was chosed. This is done
by the virtual function getActionStatistics, which is called by the getNextAction Function if an statistics object is requestet.
*/
	virtual void getActionStatistics(CStateCollection *, CAction *, CActionStatistics *) {};

public:
	///Creates a stochastic policy which can choose from the actions in "actions".
	CStochasticPolicy(CActionSet *actions, CActionDistribution *distribution);
	~CStochasticPolicy();

	/// virtual function for retrieving the action propability distribution 

	/**
	For each action in the availableActions action set, the function has to calculate the propability and write it in the double array actionValues. The function first calculates the action ratings with the function getNextAction and then calculates the action distribution with the action distribution object
*/
	virtual void getActionProbabilities(CStateCollection *state, CActionSet *availableActions, double *actionValues, CActionDataSet *actionDataSet = NULL);
/// Choses an action according the distribution from getActionPropability.
/**
First of all the available actions for the current state are calculated, and then the propabilities for this avialable
actions. Then an action is chosen from the available actions set according the distribution.
*/
	virtual CAction *getNextAction(CStateCollection *state, CActionDataSet *dataset, CActionStatistics *stat);

	/// Interface function for calculating the action ratings, has to be implemented by the subclasses
	virtual void getActionValues(CStateCollection *state, CActionSet *availableActions, double *actionValues, CActionDataSet *actionDataSet = NULL) = 0;


	virtual bool isDifferentiable() {return false;};

	virtual void getActionProbabilityGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientState);
	virtual void getActionProbabilityLnGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientState);

	/// Interface function for calculating the derivative of an action factor.
	/** 
	The function has to calculate d_actionratings(action)/dw, which is for example dQ(s,a)/dw.
	*/
	virtual void getActionGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientState);
};

/// Stochastic Policy which computes its propabilities from the Q-Values of
/**
This stochastic policy calculates its action ratings according to the given Q-Function. The getActionValues function writes the Q-Values in the actionFactors array. 
The Q-Stochastic Policies also support gradient calculation. The policy is differentiable, if the distribution and the Q-Function are differentiable. The gradient d_actionratings(action) / dw calculated in the function getActionGradient is the same as dQ(s,a)/dw. 
*/


class CQStochasticPolicy : public CStochasticPolicy
{
protected:
/// QFunction of the policy, needed for action decision
	CAbstractQFunction *qfunction;
/// returns the action statistics object from the q-function
    virtual void getActionStatistics(CStateCollection *state, CAction *action, CActionStatistics *stat);

public:
	CQStochasticPolicy(CActionSet *actions, CActionDistribution *distribution, CAbstractQFunction *qfunction);
	~CQStochasticPolicy();

	virtual void getActionValues(CStateCollection *state, CActionSet *availableActions, double *actionValues, CActionDataSet *actionDataSet = NULL);

	virtual void getActionGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientState);
	virtual bool isDifferentiable();

	virtual CAbstractQFunction *getQFunction() {return qfunction;};
};

class CQFunctionFromTransitionFunction;


/// Stochastic Policy which calculates its action from a Dynamic Model and a V-Function
/** 
The policy calculates its action ratings with a 1 or more step forward view using the dynamic model. For every action the successor state s' is calculated and then the value of that state is determined. The action rating of that action is then: Q(s,a) = R(s,a,s') + gamma * V(s'). This calculation is done by an own Q-Function class CQFunctionFromTransitionFunction. This Q-Function also supports a larger search deep than 1, the search deep can be set with the parameter "SearchDepth". Be aware that large search deeps ( > 3) have large (exponational growing) computational costs.
<p>
The policy is differentiable if the distribution is differentiable and the V-Function is differentiable. The action rating gradient can be calculated easily since the reward doesn't depend on the weights, the derivative of action a is just dV(s')/dw, where s'is the successor state when taking action a.
<p>
CVMStochasticPolicy has following Parameters:
- inherits all Parameters from the action distribution
- "SearchDepth" : number of forward search steps in the value function.
- "DiscountFactor" : gamma
*/

class CVMStochasticPolicy : public CQStochasticPolicy
{
protected:
	CStateCollectionImpl *nextState;
	CStateCollectionImpl *intermediateState;

	CAbstractVFunction *vFunction;
	CQFunctionFromTransitionFunction *qFunctionFromTransitionFunction;
	CTransitionFunction *model;
	CRewardFunction *reward;
public:
	
	CVMStochasticPolicy(CActionSet *actions, CActionDistribution *distribution, CAbstractVFunction *vFunction, CTransitionFunction *model, CRewardFunction *reward, std::list<CStateModifier *> *modifiers);
	~CVMStochasticPolicy();

	virtual void getActionGradient(CStateCollection *state, CAction *action, CActionData *data, CFeatureList *gradientState);

	virtual bool isDifferentiable();
};


#endif

