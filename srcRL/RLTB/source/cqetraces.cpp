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
#include "cqetraces.h"


#include "cqfunction.h"
#include "cvetraces.h"
#include "cvfunction.h"
#include "cfeaturefunction.h"
#include "caction.h"
#include "ril_debug.h"
#include "cutility.h"

#include <assert.h>
#include <math.h>

CAbstractQETraces::CAbstractQETraces(CAbstractQFunction *qFunction)
{
	this->qFunction = qFunction;
	
	addParameter("Lambda", 0.9);
	addParameter("DiscountFactor", 0.95);

	addParameter("ReplacingETraces", 1.0);
}

void CAbstractQETraces::setLambda(double lambda)
{
	setParameter("Lambda", lambda);
}
	
double CAbstractQETraces::getLambda()
{
	return getParameter("Lambda");
}

void CAbstractQETraces::setReplacingETraces(bool bReplace)
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

bool CAbstractQETraces::getReplacingETraces()
{
	return getParameter("ReplacingETraces") > 0.5;
}

CQETraces::CQETraces(CQFunction *qfunction) : CAbstractQETraces(qfunction)
{
	this->qExFunction = qfunction;

	vETraces = new std::list<CAbstractVETraces *>();

	CActionSet::iterator it = qExFunction->getActions()->begin();

	for (unsigned int i = 0; i < qExFunction->getNumActions(); i ++, it ++)
	{
		if ((*it) != NULL)
		{
			CAbstractVETraces *vETrace = qExFunction->getVFunction(*it)->getStandardETraces();
			addParameters(vETrace);
			vETraces->push_back(vETrace);
		}
		else
		{
			vETraces->push_back(NULL);
		}
	}
}

CQETraces::~CQETraces()
{
	std::list<CAbstractVETraces *>::iterator it = vETraces->begin();
	for (; it != vETraces->end(); it++)
	{
		if (*it != NULL)
		{
			delete (*it);
		}
	}
	
	delete vETraces;
}

void CQETraces::resetETraces()
{
	std::list<CAbstractVETraces *>::iterator it = vETraces->begin();
	for (int i = 0; it != vETraces->end(); it++, i++)
	{
		assert((*it) != NULL);
		(*it)->resetETraces();
	}
}

void CQETraces::updateETraces(CAction *action, CActionData *data)
{
	std::list<CAbstractVETraces *>::iterator it = vETraces->begin();
	int duration = 1;
	if (action->isType(MULTISTEPACTION))
	{
		if (data)
		{
			duration = dynamic_cast<CMultiStepActionData *>(data)->duration;
		}
		else
		{
			duration = action->getDuration();
		}
	}

	for (int i = 0; it != vETraces->end(); it++, i++)
	{
		assert((*it) != NULL);
		(*it)->updateETraces(duration);
	}
}


void CQETraces::addETrace(CStateCollection *state, CAction *action, double factor, CActionData *)
{
	int index = qExFunction->getActions()->getIndex(action);
	std::list<CAbstractVETraces *>::iterator it = vETraces->begin();

	for (int i = 0; it != vETraces->end(); it++, i++)
	{
		assert((*it) != NULL);
		
		if (index == i)
		{
			(*it)->addETrace(state, factor);
			break;
		}
	}
}

void CQETraces::updateQFunction(double td) 
{
	std::list<CAbstractVETraces *>::iterator it = vETraces->begin();
	CAbstractVFunction *vFunction;

	for (int i = 0; it != vETraces->end(); it++, i++)
	{
		vFunction = qExFunction->getVFunction(i);
		if ((*it) != NULL && (*it)->getVFunction() != vFunction)
		{
			delete *it;
			*it = vFunction->getStandardETraces();
		}
		
		DebugPrint('e', "ETraces Nr: %d %f\n", i, td);
		
		(*it)->updateVFunction(td);
	}
}

void CQETraces::setVETrace(CAbstractVETraces *vETrace, int index, bool bDelete)
{
	assert(qExFunction->getVFunction(index) == vETrace->getVFunction());

	std::list<CAbstractVETraces *>::iterator it = vETraces->begin();
	
	for (int i = 0; i < index; i++, it ++);
	
	if (bDelete && *it != NULL)
	{
		delete *it;
	}
	*it = vETrace;

	addParameters(vETrace);
}

CAbstractVETraces *CQETraces::getVETrace(int index)
{
	std::list<CAbstractVETraces *>::iterator it = vETraces->begin();
	
	for (int i = 0; i < index; i++, it ++);
	
	return *it;
}

void CQETraces::setReplacingETraces(bool bReplace)
{
	std::list<CAbstractVETraces *>::iterator it = vETraces->begin();
	

	for (int i = 0; it != vETraces->end(); it++, i++)
	{
		if ((*it) != NULL)
		{
			(*it)->setReplacingETraces(bReplace);
		}
	}
}



CComposedQETraces::CComposedQETraces(CComposedQFunction *qfunction) : CAbstractQETraces(qfunction)
{
	this->qCompFunction = qfunction;

	qETraces = new std::list<CAbstractQETraces *>();

	std::list<CAbstractQFunction *>::iterator it = qCompFunction->getQFunctions()->begin();

	for (int i = 0; i < qCompFunction->getNumQFunctions(); i ++, it ++)
	{
		if ((*it) != NULL)
		{
			CAbstractQETraces *qETrace = (*it)->getStandardETraces();
			addParameters(qETrace);
			qETraces->push_back(qETrace);
		}
		else
		{
			qETraces->push_back(NULL);
		}
	}
}
	
CComposedQETraces::~CComposedQETraces()
{
	std::list<CAbstractQETraces *>::iterator it = qETraces->begin();
	for (; it != qETraces->end(); it++)
	{
		if (*it != NULL)
		{
			delete (*it);
		}
	}

	delete qETraces;
}

void CComposedQETraces::resetETraces()
{
	std::list<CAbstractQETraces *>::iterator it = qETraces->begin();
	for (int i = 0; it != qETraces->end(); it++, i++)
	{
		assert((*it) != NULL);
		(*it)->resetETraces();
	}
}

void CComposedQETraces::addETrace(CStateCollection *state, CAction *action, double factor,  CActionData *data )
{
	std::list<CAbstractQETraces *>::iterator it = qETraces->begin();	
	std::list<CAbstractQFunction *>::iterator itQFunc = qCompFunction->getQFunctions()->begin();


	for (int i = 0; it != qETraces->end(); it++, i++)
	{
		assert((*it) != NULL);

		if ((*itQFunc)->getActions()->isMember(action))
		{
			(*it)->addETrace(state, action, factor, data);
			break;
		}
	}
}

void CComposedQETraces::updateETraces(CAction *action,  CActionData *data)
{
	std::list<CAbstractQETraces *>::iterator it = qETraces->begin();
	
	for (int i = 0; it != qETraces->end(); it++, i++)
	{
		assert((*it) != NULL);
		(*it)->updateETraces(action, data);
	}	
}

void CComposedQETraces::updateQFunction(double td)
{
	std::list<CAbstractQETraces *>::iterator it = qETraces->begin();
	//std::list<CAbstractQFunction *>::iterator itQFunc = qCompFunction->getQFunctions()->begin();

	//CAbstractQFunction *qFunction;

	for (; it != qETraces->end(); it++)
	{
		(*it)->updateQFunction(td);
	}
}

void CComposedQETraces::setQETrace(CAbstractQETraces *qETrace, int index, bool bDeleteOld)
{
	std::list<CAbstractQETraces *>::iterator it = qETraces->begin();

	for (int i = 0; i < index; i++, it ++);

	if (bDeleteOld && (*it) != NULL)
	{
		delete *it;
	}
	*it = qETrace;
}


CAbstractQETraces *CComposedQETraces::getQETrace(int index)
{
	std::list<CAbstractQETraces *>::iterator it = qETraces->begin();

	for (int i = 0; i < index; i++, it ++);

	return *it;
}

void CComposedQETraces::setReplacingETraces(bool bReplace)
{
	std::list<CAbstractQETraces *>::iterator it = qETraces->begin();


	for (int i = 0; it != qETraces->end(); it++, i++)
	{
		if ((*it) != NULL)
		{
			(*it)->setReplacingETraces(bReplace);
		}
	}
}
/*
void CComposedQETraces::setLambda(double lambda)
{
	this->lambda = lambda;
	std::list<CAbstractQETraces *>::iterator it = qETraces->begin();


	for (int i = 0; it != qETraces->end(); it++, i++)
	{
		if ((*it) != NULL)
		{
			(*it)->setLambda(lambda);
		}
	}
}*/


CGradientQETraces::CGradientQETraces(CGradientQFunction *qfunction) : CAbstractQETraces(qfunction)
{
	this->gradientQFunction = qfunction;

	gradient = new CFeatureList(10);
	eTrace = new CFeatureList(10, true, true);

	addParameter("ETraceTreshold", 0.001);
	addParameter("ETraceMaxListSize", 1000);


}

CGradientQETraces::~CGradientQETraces()
{
	delete gradient;
	delete eTrace;
}


void CGradientQETraces::resetETraces()
{
	eTrace->clear();
}

void CGradientQETraces::addETrace(CStateCollection *State, CAction *action, double factor, CActionData *data)
{
	gradient->clear();
	gradientQFunction->getGradient(State, action, data, gradient);

	addGradientETrace(gradient, factor);
}

void CGradientQETraces::addGradientETrace(CFeatureList *l_gradient, double factor)
{
	CFeatureList::iterator it = l_gradient->begin();

	bool replacingETraces = this->getReplacingETraces();

	for (; it != l_gradient->end(); it++)
	{
		DebugPrint('e', "%d : %f -> ",(*it)->featureIndex, eTrace->getFeatureFactor((*it)->featureIndex));

		double featureFactor = (*it)->factor * factor;

		bool signNew = featureFactor > 0;
		bool signOld = eTrace->getFeatureFactor((*it)->featureIndex) > 0;

		if (replacingETraces)
		{
			if (signNew == signOld)
			{
				if (fabs(featureFactor) > fabs(eTrace->getFeatureFactor((*it)->featureIndex)))
				{
					eTrace->set((*it)->featureIndex ,featureFactor);
				}
			}
			else
			{
				eTrace->update((*it)->featureIndex ,featureFactor);
			}
		}
		else
		{
			eTrace->update((*it)->featureIndex ,featureFactor);
		}

		DebugPrint('e', "%f\n", eTrace->getFeatureFactor((*it)->featureIndex));
	}

	int maxSize = my_round(getParameter("ETraceMaxListSize"));

	while (eTrace->size() > maxSize && maxSize > 0)
	{
		eTrace->remove(* eTrace->rbegin());
	}

}

void CGradientQETraces::updateETraces(CAction *action,  CActionData *data)
{
	CFeatureList::iterator it = eTrace->begin();

	int duration = action->getDuration();

	if (DebugIsEnabled('e'))
	{
		DebugPrint('e', "Etraces Bevore Updating: ");
		eTrace->saveASCII(DebugGetFileHandle('e'));
		DebugPrint('e',"\n");
	}

	if (action->isType(MULTISTEPACTION))
	{
		if (data)
		{
			duration = dynamic_cast<CMultiStepActionData *>(data)->duration;
		}
		else
		{
			duration = action->getDuration();
		}
	}

	int i = 0;

	double mult = getParameter("Lambda") * pow(getParameter("DiscountFactor"), duration);
	double treshold = getParameter("ETraceTreshold");

	while (it != eTrace->end())
	{
		(*it)->factor *= mult;
		if (fabs((*it)->factor) < treshold)
		{
			DebugPrint('e', "Deleting Etrace %d\n", (*it)->featureIndex);
			eTrace->remove(*it);
			
			it = eTrace->begin();
			for (int j = 0; j < i; j++, it++);

			
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
		eTrace->saveASCII(DebugGetFileHandle('e'));
		DebugPrint('e',"\n");
	}
}

void CGradientQETraces::updateQFunction(double td)
{
	DebugPrint('t', "Updating GradientQ-Function with TD %f \n", td);
	gradientQFunction->updateGradient(eTrace, td);
}

