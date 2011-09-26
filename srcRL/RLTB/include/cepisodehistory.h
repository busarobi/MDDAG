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

#ifndef C_EPISODEHISTORY_H
#define C_EPISODEHISTORY_H

#include "cagentlistener.h"
#include "chistory.h"
#include "cagentcontroller.h"
#include "cbaseobjects.h"
#include "cenvironmentmodel.h"

#include <vector>

class CEpisode;

/// Interface for all classes which are able to store episodes
/** This class is a interface for all histories containing episode objects. It provides abstract functions for 
fetching a specific episode (getEpisode) and retrieving the number of episodes (getNumEpisodes). These 2 functions 
must be implemented by all classes implementing this interface (e.g. CAgentLogger).
<p>
The class also provides to functions for simulating a certain episode (simulateEpisode) and simulating all episodes
(simulateAllEpisodes) to a specific listener. Since CAgentLogger implements this interface it can be used to simulate
the training trial to a single listener. If you use more than one listener it is recommended to use
CStoredEpisodeModel. 
@see CAgentLogger
*/

class CEpisodeHistory : virtual public CStateModifiersObject, public CStepHistory
{
protected:
	std::map<int, CEpisode*> *stepToEpisodeMap;
	std::map<CEpisode*, int> *episodeOffsetMap;
public:
	CEpisodeHistory(CStateProperties *properties, CActionSet *actions);
	virtual ~CEpisodeHistory();

/// returns the number of episodes
	virtual int getNumEpisodes() = 0;
/// returns a specific episode
	virtual CEpisode* getEpisode(int index) = 0;



	virtual int getNumSteps();
	virtual void getStep(int index, CStep *step);
	
	virtual void createStepToEpisodeMap();
};

class CEpisodeHistorySubset : public CEpisodeHistory
{
	protected:
		CEpisodeHistory *episodes;
		std::vector<int> *indices;
	public:
		CEpisodeHistorySubset(CEpisodeHistory *episodes, std::vector<int> *indices);
		virtual ~CEpisodeHistorySubset();

		/// returns the number of episodes
		virtual int getNumEpisodes();
		/// returns a specific episode
		virtual CEpisode* getEpisode(int index);

		virtual void resetData() {};
		virtual void loadData(FILE *) {};
		virtual void saveData(FILE *) {};
};

/*
/// Converts a episode history to a step history
class CEpisodeToStepHistory : public CStepHistory
{
protected:
	CEpisodeHistory *episodes;
public:

	CEpisodeToStepHistory(CEpisodeHistory *history);
	virtual ~CEpisodeToStepHistory() {};

	
};*/

/// Serves as environment model for an agent, simulating the episodes from an agent logger.
/**
The simulated environment model reproduces logged episodes for the agent, it can be used as it were a
normal environment model, so the listeners of the agent can learn from the logged data as if it is new created data. 
The environment takes an episode history and simulates the episodes in sequence. Therefore it handels pointer tto the currentEpisode 
and the number of steps already simulated in that episode. 
\par
The class also serves as agent controller, giving back the actions from the steps. The environment episode must be used
as agent controller because obviously only the actions from the episode history can be "executed".
\par
When fetching a state collection via getState(CStateCollection *), the environment model registers all states in the statecollection,
which are in the episode history and member of the state collection. These states are marked as already calculated, so only the states 
which are not stored in the history have to be recalculated.
@see CEpisodeHistory
*/

class CStoredEpisodeModel : public CEnvironmentModel, public CAgentController
{
protected:
	/// Pointer to the history
	CEpisodeHistory *history;
	/// Pointer to the current episode
	CEpisode *currentEpisode;
	
	int numEpisode;
	int numStep;

	/// Sets the state to the next step
	/** Increases the numSteps field ans sets the reset flag if the episode has ended.*/
	virtual void doNextState(CPrimitiveAction *action);
	/** Sets the currentEpisode to the next Episode in history.*/
	virtual void doResetModel();

public:
	/// Creates a simulated environment model, simulating the given history
	CStoredEpisodeModel(CEpisodeHistory *history);

	~CStoredEpisodeModel();

	virtual CEpisodeHistory* getEpisodeHistory();
	virtual void setEpisodeHistory(CEpisodeHistory *hist);

/// Gets the state with the given states properties from the history
	virtual void getState(CState *state);
/** Registers all states in the statecollection,
which are in the episode history and member of the state collection. These states are marked as already calculated, so only the states 
which are not stored in the history have to be recalculated.*/
	virtual void getState(CStateCollectionImpl *stateCollection);
/// Implementation of the agent controller interface, gets the action executed in the current step.	
	virtual CAction* getNextAction(CStateCollection *state);
};

/// Class for doing batch updates during the learning trial.
/**After each episode all the episodes from the past (which are in the given episode history) are be showed to
the listener object again. This can improve learning specially for TD-Learning algorithms.
<p> 
Each time a newEpisode event occurs the episodes in the history are shown to the given listener, so te batch update object must be added to the agents listener. This is used in connection
with an agent logger which logs the current learning trial.
*/
class CBatchEpisodeUpdate : public CSemiMDPListener
{
protected:
/// listener to show the episodes
	CSemiMDPListener *listener;
/// episode history containing all episodes
	CEpisodeHistory *logger;

	int numEpisodes;
	std::list<int> *episodeIndex;

	CActionDataSet *dataSet;
	CStep *step;
public:
/// Creates a batch update object for the specified listener with the specified episode history
	CBatchEpisodeUpdate(CSemiMDPListener *listener, CEpisodeHistory *logger, int numEpisodes, std::list<CStateModifier *> *modifiers);
	~CBatchEpisodeUpdate();

/// Simulates all epsiodes from the history to the listener
/** Just calls the simulate all episodes method of the interface CEpisodeHistory.
*/
	virtual void newEpisode();

	/// simulates the requested Episode to the listener
	virtual void simulateEpisode(int episode, CSemiMDPListener *listener);
	/// simulates all episodes to the listener
	virtual void simulateAllEpisodes(CSemiMDPListener *listener);

	/// simulates numEpisodes randomly choosen Episodes to the listener.
	void simulateNRandomEpisodes(int numEpisodes, CSemiMDPListener *listener);

};


#endif



