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

#ifndef C_HISTORY_H
#define C_HISTORY_H

#include "cbaseobjects.h"
#include "clearndataobject.h"
#include "cagentlistener.h"

class CStateCollectionImpl;
class CStateModifier;
class CStateProperties;
class CAction;
class CActionDataSet;

///Class storing a single step (State-Action-State Tuple)
class CStep : public CStateObject, public CActionObject
{
public:
	CStateCollectionImpl *oldState;
	CStateCollectionImpl *newState;

	CAction *action;
	CActionDataSet *actionData;

	CStep(CStateProperties *properties, std::list<CStateModifier*> *modifiers, CActionSet *actions);
	~CStep();
};

/// Class mantaining an unordered set of steps, which can be used for Learning
/** This class provides an interface from which you can retrieve any step from the step history. This is done
by getStep(int index, CStep *step). These function must be implemented by the subclasses.
<p>
Further the class has 2 functions for showing a semiMDP listener the single, not related steps. simulateSteps(..) simulates  "num" random steps
to the given listener, simulateAllSteps simulates the number of steps in the history to the current listener. This method can only be used for learning algorithms without etraces, since the steps shown to the listener are not related, so you have to set the Parameter "Lambda" to zero if you want to use the single steps updates for normal learners.
*/
class CStepHistory : virtual public CStateModifiersObject, virtual public CLearnDataObject, public CActionObject
{
public:

	CStepHistory(CStateProperties *properties, CActionSet *actions);
	virtual ~CStepHistory() {};

	virtual int getNumSteps() = 0;

/// virtual function, should retrieve the indexth step and writes it in the listener
	virtual void getStep(int index, CStep *step) = 0;
};


/// Class for doing Batch Updates
/**
Another possibility to improve performance is the batch update. In CBatchStepUpdate after each episode the steps from a logger can be showed to
the assigned listener again. This can improve learning specially for TD-Learning algorithms, but be careful, it can also falsify the state
transition distributions, especially if the problem doesn't have exactly the Markov property. 
The class needs obviously a listener and a assigned step history. Additionaly you have to determine the number of steps which are shown
to the listener.
*/
class CBatchStepUpdate : public CSemiMDPListener
{
protected:
	int numUpdates;
	CSemiMDPListener *listener;
	CStepHistory *steps;

	CStep *step;

	CActionDataSet *dataSet;
public:
	CBatchStepUpdate(CSemiMDPListener *listener, CStepHistory *logger, int numUpdatesPerStep, int numUpdatesPerEpisode, std::list<CStateModifier *> *modifiers);
	virtual ~CBatchStepUpdate();

	virtual void newEpisode();
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);

	///simulates num random steps to the listener
	/**
	Since the sequenced states shown to the listener are not related, a newEpisode event is send to the listener after each step.
	*/
	virtual void simulateSteps(CSemiMDPListener *listener, int num);
	///simulates "getNumSteps)" random steps to the listener
	/** Calls simulateSteps(listener, getNumSteps())*/
	virtual void simulateAllSteps(CSemiMDPListener *listener);

};

// ende

#endif



