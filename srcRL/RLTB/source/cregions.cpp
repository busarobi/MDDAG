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

#include "chierarchicbehaviours.h"
#include "cstate.h"
#include "cstateproperties.h"
#include "cregions.h"

#include "ril_debug.h"

CRegion::CRegion(CStateProperties *properties) : CStateObject(properties)
{

}

CContinuousStateRegion::CContinuousStateRegion(CStateProperties *properties, double *l_minBounds, double *l_maxBounds) : CRegion(properties)
{
	minBounds = new double[properties->getNumContinuousStates()];
	maxBounds = new double[properties->getNumContinuousStates()];

	memcpy(minBounds,l_minBounds, sizeof(double) * properties->getNumContinuousStates());
	memcpy(maxBounds,l_maxBounds, sizeof(double) * properties->getNumContinuousStates());

}


CContinuousStateRegion::~CContinuousStateRegion()
{
	delete minBounds;
	delete maxBounds;
}

bool CContinuousStateRegion::isStateInRegion(CState *state)
{
	bool isInRegion = true;
	unsigned int i = 0;
	while (isInRegion && i < properties->getNumContinuousStates())
	{
		double stateVar = state->getState(properties)->getContinuousState(i);
		if (properties->getPeriodicity(i) && maxBounds[i] < minBounds[i])
		{
			isInRegion = isInRegion && (stateVar >= minBounds[i] || stateVar <= maxBounds[i]);

		}
		else
		{
			isInRegion = isInRegion && (stateVar >= minBounds[i] && stateVar <= maxBounds[i]);
		}
		i++;
	}
	return isInRegion;
}

double CContinuousStateRegion::getDistance(CState *state)
{
	double distance = 0.0;

	if (isStateInRegion(state))
	{
		return 0.0;
	}
	for (unsigned int i = 0; i < properties->getNumContinuousStates(); i++)
	{
		double l_distance_min = state->getSingleStateDifference(i, minBounds[i]);
		double l_distance_max =  state->getSingleStateDifference(i, maxBounds[i]);

		double width = (maxBounds[i] - minBounds[i]);
		double stateWidth = (properties->getMaxValue(i) - properties->getMinValue(i));

		if (maxBounds[i] < minBounds[i])
		{
			width += stateWidth;
		}

		double inf_width = stateWidth - width;
		double l_distance = 0.0;

		if (inf_width > 0 && (l_distance_min < 0 || l_distance_max > 0))
		{
			if (fabs(l_distance_min) < fabs(l_distance_max))
			{
				l_distance += pow(l_distance_min / inf_width, 2.0);
			}
			else
			{
				l_distance += pow(l_distance_max / inf_width, 2.0);
			}
		}
		if (l_distance <= 1.0)
		{
			distance += l_distance;
		}

	}
	return sqrt(distance);
}

void CContinuousStateRegion::getRandomStateSample(CState *state)
{
	for (unsigned int i = 0; i < properties->getNumContinuousStates(); i++)
	{
		double width = (maxBounds[i] - minBounds[i]);

		if (maxBounds[i] < minBounds[i])
		{
			width += properties->getMaxValue(i) - properties->getMinValue(i);
		}
		double randState = minBounds[i] + ((double) rand()) / RAND_MAX * (width);
		state->setContinuousState(i, randState);
	}
}
