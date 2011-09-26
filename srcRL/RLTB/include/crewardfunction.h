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

#ifndef C_REWARDFUNCTION_H
#define C_REWARDFUNCTION_H

#include "cbaseobjects.h"

#include "newmat/newmat.h"

class CFeatureList;
class CAbstractVFunction;
class CFeatureVFunction;

/// Class for clalculating the reward for the learning objects
/**The reward is calculated for each learning object separately. The reward function objects have to return a double-valued reward given an S-A-S tuple, 
this is done by the function getReward. So for an own learning problem this class has to be derivated to implement the getReward function. If you need access to the
action index, you have to provide an actionset for your reward function.
<p>
Each listener who needs a reward is subclass of CSemiMDPRewardListener, which maintains a reward function object and calculates the reward for the listner. 
@see CSemiMDPRewardListener
*/

class CRewardFunction
{
public:
	virtual ~CRewardFunction() {};
/// Virtual function for calculating the reward
	virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState) = 0;
};

/// Class for calculating the reward given a feature not a state
/** Used to calculate the reward for features (integer values). Also implements the CRewardFunction interface, so it can be used
as normal reward function. When the getReward function is called with state collection objects as parameters, 
the state from the specified discritizer is chosen. This state has to be a feature or discrete state, and it gets
decomposed into his features*/

class CFeatureRewardFunction : public CRewardFunction, public CStateObject
{
protected:
	CStateProperties *discretizer;

public:
/// Creates the reward function with the specified discetizer for the states
	CFeatureRewardFunction(CStateProperties *discretizer);
	virtual ~CFeatureRewardFunction();

/// Calls getReward(CState *oldState, CAction *action, CState *newState)
	virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
/// The function to be implemented by the subclasses, has to return a reward based on discrete integer states	
	virtual double getReward(int oldState, CAction *action, int newState) = 0;
/// Calls getReward(int oldState, CAction *action, int newState) with the features from the state as argument
/** The state has to be a feature state, this state gets decomposed into his features, the reward for the features
is calculated and summed up wheighted with the feature factors.
*/
	virtual double getReward(CState *oldState, CAction *action, CState *newState);
/// Deprecated
	virtual double getReward(CFeatureList *oldState, CAction *action, CFeatureList *newState);

};

/// Reward Function that only depends on the current state
/** 
This class is the interface for all reward functions that only depend on the current (model) state. All subclasses have to implement the function getStateReward, which is called by the getReward function of the class with the newState object as argument. \par
The class also offers the function getInputDerivation where the derivation of the reward function with respect to the model state can be calculated. This is needed by some algorithm (analytical pegasus), you will have to implement this function if you want to use these algorithms. 
*/
class CStateReward : public CRewardFunction, public CStateObject
{
protected:
	CStateProperties *properties;
public:
	CStateReward(CStateProperties *properties);
	virtual ~CStateReward() {};

	virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
	
	virtual double getStateReward(CState *modelState) = 0;
	virtual void getInputDerivation(CState *, ColumnVector *) {};
};

class CZeroReward : public CRewardFunction
{
public:
	virtual double getReward(CStateCollection *, CAction *, CStateCollection *) {return 0;};
};

class CRewardFunctionFromValueFunction : public CRewardFunction
{
protected:
	CAbstractVFunction *vFunction;
	bool useNewState;
public:
	CRewardFunctionFromValueFunction(CAbstractVFunction *vFunction, bool useNewState = true);

	virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);
};

class CFeatureRewardFunctionFromValueFunction : public CFeatureRewardFunction
{
protected:
	CFeatureVFunction *vFunction;
	bool useNewState;
public:
	CFeatureRewardFunctionFromValueFunction(CStateModifier *discretizer, CFeatureVFunction *vFunction, bool useNewState = true);
	~CFeatureRewardFunctionFromValueFunction();

	virtual double getReward(int oldState, CAction *action, int newState);
};


#endif


