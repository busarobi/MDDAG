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

#include "ril_debug.h"
#include "cprioritizedsweeping.h"


#include "crewardfunction.h"
#include "ctheoreticalmodel.h"
#include "cvfunction.h"
#include "cfeaturefunction.h"
#include "cstate.h"
#include "ril_debug.h"
#include "cqfunction.h"
#include "cstatemodifier.h"
#include "cstateproperties.h"
#include "cstatecollection.h"

CPrioritizedSweeping::CPrioritizedSweeping(CFeatureQFunction *qFunc, CStateModifier *discretizer, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunction, int kSteps) : CValueIteration(qFunc, model, rewardFunction), CStateObject(discretizer)
{
	addParameter("KUpdateSteps", kSteps);
}

CPrioritizedSweeping::CPrioritizedSweeping(CFeatureVFunction *vFunc, CStateModifier *discretizer, CAbstractFeatureStochasticModel *model, CFeatureRewardFunction *rewardFunction, int kSteps) : CValueIteration(vFunc, model, rewardFunction), CStateObject(discretizer)
{
	addParameter("KUpdateSteps", kSteps);
}

CPrioritizedSweeping::~CPrioritizedSweeping()
{
	
}

void CPrioritizedSweeping::nextStep(CStateCollection *oldState, CAction *, CStateCollection *)
{
	CState *state = oldState->getState(properties);

	for (unsigned int i = 0; i < state->getNumActiveDiscreteStates(); i++)
	{
		updateFeature(state->getDiscreteState(i));
	}
	
	doUpdateStepsUntilEmptyList(my_round(getParameter("KUpdateSteps")));
}


CFeatureCalculator *CPrioritizedSweeping::getFeatureCalculator()
{
	return (CFeatureCalculator *) properties;
}

