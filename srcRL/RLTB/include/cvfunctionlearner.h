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

#ifndef C_VFUNCTIONLEARNER__H
#define C_VFUNCTIONLEARNER__H



#include "cqfunction.h"
#include "cvfunction.h"
#include "cqetraces.h"
#include "cresiduals.h"
#include "ril_debug.h"


#include "cagentlistener.h"
#include "cerrorlistener.h"
#include "cparameters.h"

class CAgentController;
class CAbstractVFunction;
class CAbstractVETraces;
class CActionDataSet;
class CGradientVFunction;
class CGradientVETraces;
class CResidualFunction;
class CResidualGradientFunction;
class CFeatureList;
class CAbstractBetaCalculator;
class CFeatureVFunction;

/// Adaptive Parameter Calculator which calculates the parameter's value from the current value of a V-Function
/**
The target value in this class is the current value of the specified V-Function, so its target value is bounded. 
For more details see the super class. 
Parameters of CAdaptiveParameterFromNStepsCalculator:
see CAdaptiveParameterBoundedValuesCalculator
*/
class CAdaptiveParameterFromValueCalculator : public CAdaptiveParameterBoundedValuesCalculator, public CSemiMDPListener
{
protected:
	CAbstractVFunction *vFunction;

	int nSteps;
	int nStepsPerUpdate;
	
	double value;
public:
	CAdaptiveParameterFromValueCalculator(CParameters *targetObject, string targetParameter, CAbstractVFunction *vFunction, int stepsPerUpdate, int functionKind, double param0, double paramScale, double targetMin, double targetMax);
	~CAdaptiveParameterFromValueCalculator();

	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState);
	virtual void onParametersChanged(){CAdaptiveParameterBoundedValuesCalculator::onParametersChanged();}; 
	
	virtual void resetCalculator();
};

/// TD Learner for Value Function learning
/**
The Value function is learned by a normal TD-Update similar to the TD-Learner for Q-Learning. The temporal difference is calculated each step with the formular td = r_t + gamma * V(s_{t+1}) - V(s_t). 
The class CVFunctionLearner uses an CVEtraces object to boost learning. The etraces update the V-Function each step with the temporal difference value, which gets multiplied by the learning rate (Parameter: "VLearningRate") before updating. Each step, the etraces are multiplied by the usual attentuation factor (lambda * gamma) and the current step is added to the etraces. When a new episode is started the etraces gets reseted. 
<p>
Value Function learner are well used in combination with a Dynamic Model for the policy (see CVMStochasticPolicy). When you use a dynamic model for your policy you learning performance will be considerably better than with Q-Learning. 
<p>
CVFunctionLearner has following Parameters:
- inherits all Parameters from the V-Function
- inherits all Parameters from the ETraces
- "VLearningRate", 0.2 : learning rate of the algorithm
- "DiscountFactor", 0.95 : discount factor of the learning problem

*/
class CVFunctionLearner : public CSemiMDPRewardListener, public CErrorSender
{
protected:
	/// learned VFunction 
	CAbstractVFunction *vFunction;
	/// Etraces of the Value Function	
	CAbstractVETraces *eTraces;

	/// are extern Etraces used?
	bool bExternETraces;

	/// adds the current state to the etrace object.
	virtual void addETraces(CStateCollection *oldState, CStateCollection *newState, int duration);

public:
	/// Creates a V-Function Learner which uses the given etraces for the V-Function.
	CVFunctionLearner(CRewardFunction *rewardFunction, CAbstractVFunction *vFunction, CAbstractVETraces *eTraces);
	/// Creates a V-Function Learner which uses the standard etraces for the V-Function.
	CVFunctionLearner(CRewardFunction *rewardFunction, CAbstractVFunction *vFunction);

	virtual ~CVFunctionLearner();

	/// Calculates the temporal difference
	/**
	The temporal difference for the given step is td = r_t + gamma * V(s_{t+1}) - V(s_t) respectively 
	td = r_t + gamma^N * V(s_{t+1}) - V(s_t) for multistep actions.
	*/
	virtual double getTemporalDifference(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
	
	/// Updates the V-Function, calls the update V-Function method of the etrace object
	/** 
	First the etraces gets multiplied by the attentuation factor (lambda * gamma)^duration, then the etrace of the current step gets added, and than the V-Function is updated by the update function of the etrace object. The update factor is td * learningrate.
	*/
	virtual void updateVFunction(CStateCollection *oldState, CStateCollection *newState, int duration, double td);

	/// Calls updateVFunction with the calculated temporal difference
	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
	
	/// Updates the V-Function for a intermediate step (only for hierarchic MDP's)
	/**
	Since the intermediate steps aren't doublely member of the hierarchic episode they need special treatment for etraces.
	The state of the intermediate step is added to the ETraces object as usual, but the attenutuation of all other etraces is canceled and the V-Function isn’t updated with the whole ETraces object, only the current V-Value of the intermediate state is updated. 
	This is done because the intermediate step isn't directly reachable for the past states and update all intermediate steps via etraces would falsify the V-Values since the same step gets updates several times.
	*/
	virtual void intermediateStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
	
	/// Resets the etraces
	virtual void newEpisode();
	/// Returns the used V-Function
	CAbstractVFunction *getVFunction();

	double getLearningRate();
	void setLearningRate(double learningRate);
	/// Returns the used ETraces for the VFunction
	CAbstractVETraces *getVETraces();
};

class CVFunctionGradientLearner : public CVFunctionLearner
{
protected:
	CResidualFunction *residual;
	CResidualGradientFunction *residualGradientFunction;

	CGradientVFunction *gradientVFunction;
	CGradientVETraces *gradientETraces;

	CFeatureList *oldGradient;
	CFeatureList *newGradient;
	CFeatureList *residualGradient;

	virtual void addETraces(CStateCollection *oldState, CStateCollection *newState, int duration);
public:
	CVFunctionGradientLearner(CRewardFunction *rewardFunction, CGradientVFunction *vFunction, CResidualFunction *residual, CResidualGradientFunction *residualGradientFunction);

	~CVFunctionGradientLearner();

	virtual double getTemporalDifference(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
};

class CVFunctionResidualLearner : public CVFunctionGradientLearner
{
protected:
	CGradientVETraces *residualGradientTraces;
	CGradientVETraces *directGradientTraces;

	CGradientVETraces *residualETraces;

	CAbstractBetaCalculator *betaCalculator;

	virtual void addETraces(CStateCollection *oldState, CStateCollection *newState, int duration, double td);

public:
	CVFunctionResidualLearner(CRewardFunction *rewardFunction, CGradientVFunction *vfunction, CResidualFunction *residual, CResidualGradientFunction *residualGradient, CAbstractBetaCalculator *betaCalc);

	~CVFunctionResidualLearner();

	virtual void updateVFunction(CStateCollection *oldState, CStateCollection *newState, int duration, double td);
	
	virtual void newEpisode();

	CGradientVETraces *getResidualETraces() {return residualETraces;};
};

class CVAverageTDErrorLearner : public CErrorListener, public CStateObject
{
	protected:
		double updateRate;
			
		CFeatureVFunction *averageErrorFunction;
	public:
		CVAverageTDErrorLearner(CFeatureVFunction *averageErrorFunction, double updateRate);
		virtual ~CVAverageTDErrorLearner();
		
		virtual void receiveError(double error, CStateCollection *state, CAction *action, CActionData *data = NULL);	
		
		virtual void onParametersChanged();
};

class CVAverageTDVarianceLearner : public CVAverageTDErrorLearner
{
	public:
		CVAverageTDVarianceLearner(CFeatureVFunction *averageErrorFunction, double updateRate);
		virtual ~CVAverageTDVarianceLearner();
		
		virtual void receiveError(double error, CStateCollection *state, CAction *action, CActionData *data = NULL);	
};


#endif

