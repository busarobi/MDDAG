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

#ifndef CSEMIMDPLISTENER_H
#define CSEMIMDPLISTENER_H

#include "cparameters.h"

class CAction;
class CState;
class CStateCollection;
class CRewardFunction;

/// Interface for all SemiMDP Listeners. 
/**This class is the base class of all Learning and Logging objects. If the listeners get added to a CSemiMarkovDecisionProcess the listener 
gets informed about all Steps from the SMDP and wether to start a new Episode.
<p>
There are 3 different kind of events which can be sent to the Listener:
	- nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState). The Listener gets S-A-S Tuple of
	the current step. Usually this is the only data a learning algorithm needs (including the reward which ca be calculated from the S-A-S tuple).
	- intermediateStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState). Only needed for Hierarchical Reinforcement Learning. The Listener gets the same data as in nextStep, but instead of a double step, the S-A-S tuple comes from an intermediate step from an extended Action (see CExtendedAction for 
	Details about intermediate steps). The difference is made because intermediate steps has to be treated differently in some cases (e.g. ETraces)
	- newEpisode(). Indicates that a new Episode has startet.

One or more of this event-functions should be implemented by all subclasses.
The class is also subclass of CParameterObject, so the paramters of the listeners can be set through that interface
@see CSemiMDPSender
@see CSemiMarkovDecisionProcess
@see CParamterObject
*/
class CSemiMDPListener : virtual public CParameterObject
{
public:
	bool enabled;
	
	CSemiMDPListener() {enabled = true;};

	/// sends the Listener the S-A-S tuple from a new step
	virtual void nextStep(CStateCollection *, CAction *, CStateCollection *) {};
	/// sends the Listener the S-A-S tuple from a indermediate step
	virtual void intermediateStep(CStateCollection *, CAction *, CStateCollection *) {};
	/// tells the Listener that a new Episode has startet.
	virtual void newEpisode() {};
};


///Represents SMDP Listener which also need a reward
/** The CSemiMDPRewardListener maintains a reward function. With this reward function, each time a nextStep or an
intermediateStep event occurs the listener can calculate the reward and then he calls the specific abstract event function with the
S-A-R(eward)-S tuple. 
*/
class CSemiMDPRewardListener : public CSemiMDPListener
{
protected:
/// reward function for reward calculation
	CRewardFunction *semiMDPRewardFunction;

public:
/** @param semiMDPRewardFunction reward function for reward calculation*/
	CSemiMDPRewardListener(CRewardFunction *semiMDPRewardFunction);

/// Calculates the reward and then calls nextStep(...) with the reward as additional argument.
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);
/// virtual function, to be implemented by subclass
	virtual void nextStep(CStateCollection *, CAction *, double , CStateCollection *) {};

/// Calculates the reward and then calls intermediateStep(...) with the reward as additional argument.
	virtual void intermediateStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);
/// virtual function, to be implemented by subclass
	virtual void intermediateStep(CStateCollection *, CAction *, double , CStateCollection *) {};

	void setRewardFunction(CRewardFunction *semiMDPRewardFunction);
	CRewardFunction *getRewardFunction();
};

/// Adaptive Parameter Calculator which calculates the parameter's value from the number of learning steps
/**
The target value in this class is the number of learning steps, so its target value is unbounded. The target value gets resetted to 0 if a new learning trial has started.
This adaptive parameter has to be added to the agent's listener list in order to count the number of steps.
For more details see the super class. 
Parameters of CAdaptiveParameterFromNStepsCalculator:
see CAdaptiveParameterUnBoundedValuesCalculator
*/
class CAdaptiveParameterFromNStepsCalculator : public CAdaptiveParameterUnBoundedValuesCalculator, public CSemiMDPListener
{
protected:
	int targetValue;
	int nStepsPerUpdate;
public:
	CAdaptiveParameterFromNStepsCalculator(CParameters *targetObject, string targetParameter, int nStepsPerUpdate, int functionKind, double param0, double paramScale, double targetOffset, double targetScale);
	virtual ~CAdaptiveParameterFromNStepsCalculator();

	virtual void nextStep(CStateCollection *, CAction *, CStateCollection *);
	virtual void onParametersChanged(){CAdaptiveParameterUnBoundedValuesCalculator::onParametersChanged();}; 
	
	virtual void resetCalculator();
};

/// Adaptive Parameter Calculator which calculates the parameter's value from the number of learning episodes
/**
The target value in this class is the number of learning episodes, so its target value is unbounded. The target value gets resetted to 0 if a new learning trial has started.
This adaptive parameter has to be added to the agent's listener list in order to count the number of episodes.
For more details see the super class. 
Parameters of CAdaptiveParameterFromNStepsCalculator:
see CAdaptiveParameterUnBoundedValuesCalculator
*/
class CAdaptiveParameterFromNEpisodesCalculator : public CAdaptiveParameterUnBoundedValuesCalculator, public CSemiMDPListener
{
protected:
	int targetValue;
public:
	CAdaptiveParameterFromNEpisodesCalculator(CParameters *targetObject, string targetParameter, int functionKind, double param0, double paramScale, double targetOffset, double targetScale);
	virtual ~CAdaptiveParameterFromNEpisodesCalculator();

	virtual void newEpisode();
	virtual void onParametersChanged(){CAdaptiveParameterUnBoundedValuesCalculator::onParametersChanged();}; 

	virtual void resetCalculator();
};


/// Adaptive Parameter Calculator which calculates the parameter's value from the current average reward
/**
The target value in this class is the current average reward. The target value gets resetted the minimim expected reward if a new learning trial has started.
This adaptive parameter has to be added to the agent's listener list in order to calculate the average reward. The average reward is calculated dynamically with the formular averagereward_t+1 = averagereward_t * alpha + reward_t+1 * (1 - alpha). Alpha can be set with the parameter "APRewardUpdateRate" and defines the update rate of the average reward. Alpha should be choosen close to 0.99 to get good results. The average reward is not resetted when a new episode begins.
For more details see the super class. 
Parameters of CAdaptiveParameterFromNStepsCalculator:
- "APRewardUpdateRate": Update rate for the average reward.
see CAdaptiveParameterBoundedValuesCalculator
*/
class CAdaptiveParameterFromAverageRewardCalculator : public CAdaptiveParameterBoundedValuesCalculator, public CSemiMDPRewardListener
{
protected:
	double alpha;
	double targetValue;
	int nSteps;
	int nStepsPerUpdate;
public:
	CAdaptiveParameterFromAverageRewardCalculator(CParameters *targetObject, string targetParameter, CRewardFunction *reward, int nStepsPerUpdate, int functionKind, double paramMin, double paramMax, double targetMin, double targetMax, double alpha);
	~CAdaptiveParameterFromAverageRewardCalculator();

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState);
	virtual void onParametersChanged(); 
	
	virtual void resetCalculator();
};

#endif // CSEMIMDPLISTENER_H

