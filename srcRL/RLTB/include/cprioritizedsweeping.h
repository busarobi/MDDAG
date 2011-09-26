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

#ifndef C_PRIORITZEDSWEEPING__H
#define C_PRIORITZEDSWEEPING__H

#include "cdynamicprogramming.h"
#include "cbaseobjects.h"
#include "cagentlistener.h"

class CFeatureQFunction;
class CStateModifier;
class CAbstractFeatureStochasticModel;
class CFeatureRewardFunction;
class CFeatureVFunction;
class CFeatureCalculator;


#include <map>
/// class for model based Prioritized Sweeping
/**
Prioritized Sweeping is used if the model is learned during the training trial and is very similiar to value iteration. 
Since it would be too complex to do value iteration each time the model changes, only the first k states from the priority list are updated each step, starting with
the current features. The Prioritized Sweeping class is subclass of CValueIteration, the only extension is that, each time a nextStep event occurs, the current features are updated (and so the backward states of the current features are added to the priority list) and than the states from the list are updated k times.
Since it is subclass of CValueIteration it provides the full functionality for state updates. The prioritized sweeping class always learns the value function of the greedy policy, so its the optimal value function.
<p>
The class only provides Q-Function learning because the Q-Function is needed for the policies. Additionally it takes a model (CAbstractFeatureStochasticModel) and a feature reward function as parameters.
The model (and the reward function, if it is a reward model itself) has to be added to the agents listener list before the prioritized sweeping algorithm is added.
*/

class CPrioritizedSweeping : public CSemiMDPListener, public CValueIteration, public CStateObject
{
protected:
/// kSteps updates are done each step.
	int kSteps;
			
public:

/**
The class only provides Q-Function learning because the Q-Function is needed for the policies. Additionally it takes a model (CAbstractFeatureStochasticModel) and a feature reward function as parameters.
The model (and the reward function, if it is a reward model itself) has to be added to the agents listener list before the prioritized sweeping algorithm is added.
*/
	CPrioritizedSweeping(CFeatureQFunction *qFunction, CStateModifier *discretizer, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunction, int kSteps);

	CPrioritizedSweeping(CFeatureVFunction *vFunction, CStateModifier *discretizer, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunction, int kSteps);

	virtual ~CPrioritizedSweeping();

/// Updates the features of the old State
/** Retrieves the state from the statecollection (with the given modifier pointer from the constructor), and updates each feature.
So the backwards states priorities get updated as well. After that, the first kSteps states in the list are updated.*/
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState);

	CFeatureCalculator *getFeatureCalculator();
};

#endif

