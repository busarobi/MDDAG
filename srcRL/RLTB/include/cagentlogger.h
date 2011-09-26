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

#ifndef CAGENTLOGGER_H
#define CAGENTLOGGER_H

#include "cagentlistener.h"
#include "cepisodehistory.h"
#include "cbaseobjects.h"

#include <stdio.h>
#include <list>

class CEpisode;
class CStateModifier;
/// Class for logging a whole training trial
/**The agent logger (CAgentLogger) objects can store whole training trials. CAgentLogger is a subclass of CSemiMDPListener in order to get the data from
the agent. The agent logger uses a list of episode objects, so you can retrieve whole episodes from the logger. The curent episode isn't in that list and can 
only be referenced by the method getCurrentEpisode. From the episodes you can again retrieve the single states. You can set an auto-save file where every Episode gets saved automatically when it is finished. You can also
set the number of episodes the logger should hold in memory (both parameters are set in the contructor). If the episodes exceeds that number, the oldest episodes get discarded. holdMemory of -1 means that all episodes are
hold in the memory.
<p>
The Agentlogger stores only those states which's state modifier objects (i.e. the properties) are in the state modifier list of the logger.

@see CEpisode
*/

class CAgentLogger: public CSemiMDPListener, public CEpisodeHistory
{
protected:
	/// autosave file name
    char filename[512];
	/// autosave file
    FILE* file;

	char loadFileName[512];
	std::list<CStateModifier *> *loadModifiers;
	/// number of episodes to hold in memory, if set to -1 all episodes are held in memory
	int holdMemory;

	/// list of the episodes
	std::list<CEpisode *> *episodes;
	/// pointer to the currentepisode
	CEpisode *currentEpisode;
	
	void init();
public:
	/// Creates an agent logger and sets the autosave file 
	CAgentLogger(CStateProperties *model, CActionSet *actions, char* autoSavefile, int holdMemory);
	/// Creates an agent logger with no autosave file and all episodes held in memory.
	CAgentLogger(CStateProperties *model, CActionSet *actions);
	/// Loads an training trial from a binary file. The modifiers must be the same as the modifiers used when the trial was saved.
	CAgentLogger(char *loadFile, CStateProperties *model, CActionSet *actions, std::list<CStateModifier *> *modifiers);

	virtual ~CAgentLogger();

/// Stores the step in the current episode
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);
/// Stores the current Episode in the episode list and creates a new current Episode object.
	virtual void newEpisode();

/// sets auto save file, if set to NULL no auto save file is used
	void setAutoSaveFile(char *filename);

/// saves the whole trial to a binary file
	virtual void saveBIN(FILE *stream);
/// saves the whole trial to a text file
	virtual void saveData(FILE *stream);
/// loads a training trial from a binary file
/** if episodes is set to a value greater 0, only the first "episode" episodes are loaded*/
	void loadBIN(FILE *stream, std::list<CStateModifier *> *modifiers, int episodes = -1);
/// loads a training trial from a text file
/** if episodes is set to a value greater 0, only the first "episode" episodes are loaded*/
	virtual void loadData(FILE *stream, int episodes = -1);
	virtual void loadData(FILE *stream);

/// returns number of Episodes in the episode list (so exclusive the current Episode)
	virtual int getNumEpisodes();

/// adds a state modifier to the current episode object.
	virtual void addStateModifier(CStateModifier *modifier);
/// removes a state modifier to the current episode object.
	virtual void removeStateModifier(CStateModifier *modifier);

/// returns a pointer to the current episode.
	virtual CEpisode* getCurrentEpisode();

/// returns the index th episode.
	virtual CEpisode* getEpisode(int index);

/// clears the autosave file
	void clearAutoSaveFile();
	
	void setLoadDataFile(char *loadData, std::list<CStateModifier *> *modifiers = NULL);

/// Removes all Episodes from memory

	virtual void resetData();
};

/// This Class writes each step and start of a new episode in readable form to a file
/** For each state the old state, action, reward and newstate is written to the specified file.
For the states only the specified state is chossen from the state colection. This class can be used for 
error tracking of the model or reward model and debugging.
*/

class CEpisodeOutput : public CSemiMDPRewardListener, public CActionObject, public CStateObject
{
protected:
	FILE *stream;

	int nEpisodes;
	int nSteps;

public:
	CEpisodeOutput(CStateProperties *featCalc, CRewardFunction *rewardFunction, CActionSet *actions, FILE *output); 
	virtual ~CEpisodeOutput();

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
	virtual void intermediateStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
	virtual void newEpisode();
};

class CEpisodeMatlabOutput : public CSemiMDPRewardListener, public CActionObject, public CStateObject
{
protected:
	FILE *stream;

	
	int nSteps;

public:
	int nEpisodes;
	
	CEpisodeMatlabOutput(CStateProperties *featCalc, CRewardFunction *rewardFunction, CActionSet *actions, FILE *output); 
	virtual ~CEpisodeMatlabOutput();

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
	virtual void newEpisode();
	
	void setOutputFile(FILE *stream);
};

/// Class writes only the steps in which the specified state changes in readable form to a file
/** Does the same as CEpisodeOutput, but only for steps in which the state changes.
*/

class CEpisodeOutputStateChanged : public CEpisodeOutput
{
public:
	CEpisodeOutputStateChanged(CStateProperties *featCalc, CRewardFunction *rewardFunction, CActionSet *actions, FILE *output); 
	virtual ~CEpisodeOutputStateChanged() {};

	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState);
};

class CStateOutput : public CSemiMDPListener, public CStateObject
{
protected:
	FILE *stream;

public:
	CStateOutput(CStateProperties *featCalc, FILE *output); 
	virtual ~CStateOutput();

	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);
};

class CActionOutput : public CSemiMDPListener, public CActionObject
{
protected:
	FILE *stream;

public:
	CActionOutput(CActionSet *actions, FILE *output); 
	virtual ~CActionOutput();

	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);
};

#endif
