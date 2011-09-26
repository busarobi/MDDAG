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

#ifndef C_EXPLORATION__H
#define C_EXPLORATION__H

#include "cagentlistener.h"
#include "cqfunction.h"
#include "cpolicies.h"




class CGradientVFunction;
class CFeatureVFunction;
class CStateCollection;
class CAction;
class CActionData;

class CAbstractVFunction;
class CAbstractQFunction;

class CVisitStateCounter : public CSemiMDPListener
{
protected:
	CGradientVFunction *visits;
	double *weights;
	int steps;

	virtual void doDecay(double decay);
public:
	CVisitStateCounter(CFeatureVFunction *visits, double decay = 1.0);
	virtual ~CVisitStateCounter();

	virtual void nextStep(CStateCollection *state, CAction *action, CStateCollection *nextState);
	virtual void newEpisode();
};

class CVisitStateActionCounter : public CSemiMDPListener
{
protected:
	CGradientQFunction *visits;
	double *weights;
	int steps;

	virtual void doDecay(double decay);

public:
	CVisitStateActionCounter(CFeatureQFunction *visits, double decay = 1.0);
	virtual ~CVisitStateActionCounter();

	virtual void nextStep(CStateCollection *state, CAction *action, CStateCollection *nextState);
	virtual void newEpisode();
};

class CVisitStateActionEstimator : public CVisitStateCounter
{
protected:
	CGradientQFunction *actionVisits;

	virtual void doDecay(double decay);

public:
	CVisitStateActionEstimator(CFeatureVFunction *stateVisits, CFeatureQFunction *actionVisits, double decay = 1.0);
	virtual ~CVisitStateActionEstimator();

	virtual void nextStep(CStateCollection *state, CAction *action, CStateCollection *nextState);
	virtual void newEpisode();
};

class CExplorationQFunction : public CAbstractQFunction
{
protected:
	CAbstractVFunction *stateVisitCounter;
	CAbstractQFunction *actionVisitCounter;

public:
	CExplorationQFunction(CAbstractVFunction *stateVisitCounter, CAbstractQFunction *actionVisitCounter);

	virtual ~CExplorationQFunction();

	virtual void updateValue(CStateCollection *state, CAction *action, double td, CActionData *data = NULL);
	virtual void setValue(CStateCollection *state, CAction *action, double qValue, CActionData *data = NULL); 
	virtual double getValue(CStateCollection *state, CAction *action, CActionData *data = NULL);

	virtual CAbstractQETraces *getStandardETraces();
};


class CQStochasticExplorationPolicy : public CQStochasticPolicy
{
protected:
	CAbstractQFunction *explorationFunction;

	double *explorationValues;

public:
	CQStochasticExplorationPolicy(CActionSet *actions, CActionDistribution *distribution, CAbstractQFunction *qFunctoin, CAbstractQFunction *explorationFunction, double explorationFactor);
	~CQStochasticExplorationPolicy();

	virtual void getActionValues(CStateCollection *state, CActionSet *availableActions, double *actionValues, CActionDataSet *actionDataSet = NULL);

	virtual CAbstractQFunction *getExplorationQFunction() {return explorationFunction;};
};

class CSelectiveExplorationCalculator : public CSemiMDPListener
{
protected:
	CQStochasticExplorationPolicy *explorationPolicy;
	
	double attention;
public:
	CSelectiveExplorationCalculator(CQStochasticExplorationPolicy *explorationFunction);
	virtual ~CSelectiveExplorationCalculator();

	virtual void nextStep(CStateCollection *state, CAction *action, CStateCollection *nextState);
	virtual void newEpisode();
};


#endif

