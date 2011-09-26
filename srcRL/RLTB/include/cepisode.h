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

#ifndef C_EPISODE_H
#define C_EPISODE_H

#include <vector>
#include <vector>
#include <map>

#include "chistory.h"
#include "cagentlistener.h"

class CState;
class CStateProperties;
class CStateCollection;
class CStateCollectionList;
class CStateModifier;
class CStateList;
class CAction;
class CActionList;
class CActionDataSet;

/// Class for logging a single Episode
/**
The episodes objects (CEpisode) can store only one episode. The episode stores numSteps actions and numSteps + 1 states, therefore it uses a CActionList and a CStateCollectionList object. The episodes are also listeners, so you can add them to the agent’s listeners list and they will log the current episode. When a newEpisode event occurs the episode object is cleared. The agent logger uses a list CEpisode objects to log the whole training trial. There are always numSteps + 1 states and numSteps actions in the lists (except numSteps = 0, then both lists are empty).
The episode stores only those states which's state modifier objects (i.e. the properties) are in the state modifier list of the logger. For retrieving state collections or action objects, the methods from the CStateCollectionList and CActionList objects are used.
The episode implements the interface CStepHistory, so you can retrieve random steps for learning from any episode. This is an alternative way of updating you Q or V Functions.
@see CEpisodeHistory
@see CAgentLogger
*/

class CEpisode : public CSemiMDPListener, public CStepHistory
{
protected:

/// clear Episode when a newEpisode event occurs, default = true
	bool autoNewEpisode;

/// state collection list for the numSteps + 1 states
	CStateCollectionList *stateCollectionList;
/// action list for the numSteps actions
	CActionList *actionList;
public:
/// Creates an CEpisode object, the base state of the collection list has the properties "properties".
	CEpisode(CStateProperties *properties, CActionSet *actions, bool autoNewEpisode = true);
/// Creates an CEpisode object, the base state of the collection list has the properties "properties".
/** initializes the collection list with the modifiers */
	CEpisode(CStateProperties *properties, CActionSet *actions, std::list<CStateModifier *> *modifiers, bool autoNewEpisode = true);
	virtual ~CEpisode();

/** Stores the newState and the action, if it was the first step, the oldState object is also saved (before newState).
*/
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *newState );

/// if autoNewEpisode is true, the episode gets cleared
	virtual void newEpisode();

	CStateList *getStateList(CStateProperties *properties);

/// get a state collection from the Episode, just calls the equivalent function of the CStateCollectionList object
	void getStateCollection(int index, CStateCollectionImpl *stateCollection);
/// get a state from the Episode, just calls the equivalent function of the CStateCollectionList object
	void getState(int index, CState *state);

/// add state modifier to the episodes state collection list
	virtual void addStateModifier(CStateModifier *modifier);
/// remove state modifier to the episodes state collection list
	virtual void removeStateModifier(CStateModifier *modifier);

/// get a action from the Episode, just calls the equivalent function of the CActionList object
	CAction *getAction(unsigned int num, CActionDataSet *dataSet = NULL);

/// Save as a binary
/** Uses the save functions from the state collection list and the action list object. */
	void saveBIN(FILE *stream);
/// Load from a binary
/** Uses the load functions from the state collection list and the action list object. */
	void loadBIN(FILE *stream);

/// Save as text
/** Uses the save functions from the state collection list and the action list object. */
	virtual void saveData(FILE *stream);
/// Load from a text
/** Uses the load functions from the state collection list and the action list object. */
	virtual void loadData(FILE *stream);

/// returns the number of steps logged by the episode.
	virtual int getNumSteps();

/// returns the num th step.
	virtual void getStep(int num, CStep *step);

	virtual int getNumStateCollections();

	virtual void resetData();
};




#endif


