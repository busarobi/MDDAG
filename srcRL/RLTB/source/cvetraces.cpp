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
#include "cvetraces.h"

#include "cfeaturefunction.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "cstatemodifier.h"
#include "cutility.h"
#include "cvfunction.h"


#include <assert.h>
#include <math.h>

CAbstractVETraces::CAbstractVETraces(CAbstractVFunction *vFunction)
{
	this->vFunction	= vFunction;

	addParameter("Lambda",0.9);
	addParameter("DiscountFactor", 0.95);
	addParameter("ETraceTreshold", 0.001);

	addParameter("ReplacingETraces", 1.0);
}

void CAbstractVETraces::setReplacingETraces(bool bReplace)
{
	if (bReplace)
	{
		setParameter("ReplacingETraces", 1.0);
	}
	else
	{
		setParameter("ReplacingETraces", 0.0);

	}
}

bool CAbstractVETraces::getReplacingETraces()
{
	return getParameter("ReplacingETraces") > 0.5;
}

void CAbstractVETraces::setLambda(double lambda)
{
	setParameter("Lambda", lambda);
}
	
double CAbstractVETraces::getLambda()
{
	return getParameter("Lambda");
}

void CAbstractVETraces::setTreshold(double treshold)
{
	setParameter("ETraceTreshold", treshold);
}
	
double CAbstractVETraces::getTreshold()
{
	return getParameter("ETraceTreshold");
}

CAbstractVFunction *CAbstractVETraces::getVFunction()
{
	return vFunction;
}

CStateVETraces::CStateVETraces(CAbstractVFunction *vFunction, CStateProperties *modelState, std::list<CStateModifier *> *modifiers) : CAbstractVETraces(vFunction)
{
	setParameter("ETraceTreshold", 0.05);
	addParameter("ETraceMaxListSize", 10);

	if (modifiers != NULL)
	{
		eTraceStates = new CStateCollectionList(modelState, modifiers);
		bufState = new CStateCollectionImpl(modelState, modifiers);
	}
	else
	{
		eTraceStates = new CStateCollectionList(modelState);
		bufState = new CStateCollectionImpl(modelState);
	}

	eTraces = new std::list<double>();
}

CStateVETraces::~CStateVETraces()
{
	delete eTraceStates;
	delete bufState;
	delete eTraces;
}

void CStateVETraces::resetETraces()
{
	eTraces->clear();
	eTraceStates->clearStateLists();
}

void CStateVETraces::updateETraces(int duration)
{
	double mult = getParameter("Lambda") * pow(getParameter("DiscountFactor"), duration);
	std::list<double>::iterator eIt = eTraces->begin();
	double treshold = getParameter("ETraceTreshold");

	for (; eIt != eTraces->end(); eIt ++)
	{
		(*eIt) = (*eIt) * mult;
		if (*eIt < treshold)
		{
			eTraces->erase(eIt, eTraces->end());
			break;
		}
	}
}


void CStateVETraces::addETrace(CStateCollection *state, double factor)
{	
	eTraceStates->addStateCollection(state);
	eTraces->push_front(factor);
}

void CStateVETraces::updateVFunction(double td)
{
	std::list<double>::iterator evalue = eTraces->begin();

	
	for (int state = 0; evalue != eTraces->end(); evalue ++)
	{
		eTraceStates->getStateCollection(eTraceStates->getNumStateCollections() - state, bufState);
		vFunction->updateValue(bufState, (*evalue) * td);
	}
}


CGradientVETraces::CGradientVETraces(CGradientVFunction *gradientVFunction) : CAbstractVETraces(gradientVFunction)
{
	eFeatures = new CFeatureList(10, true, true);

	tmpList = new CFeatureList();

	addParameter("ETraceMaxListSize", 1000);

	this->gradientVFunction = gradientVFunction;
}

CGradientVETraces::~CGradientVETraces()
{
	delete eFeatures;
	delete tmpList;
}

void CGradientVETraces::resetETraces()
{
	eFeatures->clear();	
}

void CGradientVETraces::updateETraces(int duration)
{
	double lambda = getParameter("Lambda");
	
	double mult = lambda * pow(getParameter("DiscountFactor"), duration);

	multETraces(mult);	
}


void CGradientVETraces::addETrace(CStateCollection *State, double factor)
{
	tmpList->clear();
	gradientVFunction->getGradient(State, tmpList);

	addGradientETrace(tmpList, factor);

}

void CGradientVETraces::multETraces(double mult)
{
	CFeatureList::iterator it = eFeatures->begin();
	int i = 0;
	double treshold = getParameter("ETraceTreshold");

	if (DebugIsEnabled('e'))
	{
		DebugPrint('e', "Etraces Bevore Updating (factor: %f): ", mult);
		eFeatures->saveASCII(DebugGetFileHandle('e'));
		DebugPrint('e',"\n");
	}

	while (it != eFeatures->end())
	{
		(*it)->factor *= mult;
		if (fabs((*it)->factor) < treshold)
		{
			eFeatures->remove(*it);
			it = eFeatures->begin();
			for (int j = 0; j < i; j++, it++);

			//printf("Deleting Etrace \n");
		}
		else
		{
			i++;
			it++;
		}
	}

	if (DebugIsEnabled('e'))
	{
		DebugPrint('e', "Etraces After Updating: ");
		eFeatures->saveASCII(DebugGetFileHandle('e'));
		DebugPrint('e',"\n");
	}
}

void CGradientVETraces::addGradientETrace(CFeatureList *gradient, double factor)
{
	CFeatureList::iterator it = gradient->begin();
	DebugPrint('e', "Adding Etraces:\n");
	int maxListSize = my_round(getParameter("ETraceMaxListSize"));

	bool replacing = this->getReplacingETraces();
	for (; it != gradient->end(); it++)
	{
		DebugPrint('e', "%d : %f -> ",(*it)->featureIndex, eFeatures->getFeatureFactor((*it)->featureIndex));

		double featureFactor = (*it)->factor * factor;

		bool signNew = featureFactor > 0;
		bool signOld = eFeatures->getFeatureFactor((*it)->featureIndex) > 0;

		if (replacing)
		{
			if (signNew == signOld)
			{
				if (fabs(featureFactor) > fabs(eFeatures->getFeatureFactor((*it)->featureIndex)))
				{
					eFeatures->set((*it)->featureIndex ,featureFactor);
				}
			}
			else
			{
				eFeatures->update((*it)->featureIndex ,featureFactor);
			}
		}
		else
		{
			eFeatures->update((*it)->featureIndex ,featureFactor);
		}
		DebugPrint('e', "%f\n", eFeatures->getFeatureFactor((*it)->featureIndex));
	}

	while (eFeatures->size() > maxListSize)
	{
		eFeatures->remove(*eFeatures->rbegin());
	}
}

	
void CGradientVETraces::updateVFunction(double td)
{
	gradientVFunction->updateGradient(eFeatures, td);
}

CFeatureList* CGradientVETraces::getGradientETraces()
{
	return eFeatures;	
}


CFeatureVETraces::CFeatureVETraces(CFeatureVFunction *featureVFunction) : CGradientVETraces(featureVFunction)
{
	this->featureVFunction = featureVFunction;

	this->featureProperties = featureVFunction->getStateProperties();

}



CFeatureVETraces::CFeatureVETraces(CFeatureVFunction *featureVFunction, CStateProperties *featureProperties) : CGradientVETraces(featureVFunction)
{
	this->featureVFunction = featureVFunction;

	this->featureProperties = featureProperties;
}


void CFeatureVETraces::addETrace(CStateCollection *stateCol, double factor)
{
	bool replacing = this->getReplacingETraces();

	if (stateCol != NULL)
	{
		int maxListSize = my_round(getParameter("ETraceMaxListSize"));

		DebugPrint('e', "Adding Etraces:\n");

		CState *state = stateCol->getState(featureProperties);

		double featureFactor = 0.0;

		for (unsigned int i = 0; i < state->getNumActiveDiscreteStates(); i++)
		{
			int type = state->getStateProperties()->getType() & (FEATURESTATE | DISCRETESTATE);
			switch (type)
			{
			case DISCRETESTATE:
				{
					featureFactor = 1.0 * factor;
					break;
				}
			case FEATURESTATE:
				{
					featureFactor = state->getContinuousState(i) * factor;
					break;
				}
			default:
				{
					featureFactor = 1.0 * factor;
				}
			}
			bool signNew = featureFactor > 0;
			bool signOld = eFeatures->getFeatureFactor(state->getDiscreteState(i)) > 0;

			if (replacing)
			{
				if (signNew == signOld)
				{
					if (fabs(featureFactor) < fabs(eFeatures->getFeatureFactor(state->getDiscreteState(i))))
					{
						featureFactor = eFeatures->getFeatureFactor(state->getDiscreteState(i));
					}
				}
				else
				{
					featureFactor += eFeatures->getFeatureFactor(state->getDiscreteState(i));
				}
			}
			else
			{
				featureFactor += eFeatures->getFeatureFactor(state->getDiscreteState(i));
			}
			DebugPrint('e', "%d: %f -> ", state->getDiscreteState(i), eFeatures->getFeatureFactor(state->getDiscreteState(i)));
			eFeatures->set(state->getDiscreteState(i),featureFactor);
			DebugPrint('e', "%f\n", eFeatures->getFeatureFactor(state->getDiscreteState(i)));
			
		}
		while (eFeatures->size() > maxListSize)
		{
			eFeatures->remove(*eFeatures->rbegin());
		}
	}
}

