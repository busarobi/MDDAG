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

#include "cagentlogger.h"
#include "ccontinuousactions.h"
#include "cepisode.h"
#include "chistory.h"
#include "crewardfunction.h"
#include "cstateproperties.h"
#include "cstatecollection.h"
#include "caction.h"
#include "ril_debug.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>


CAgentLogger::CAgentLogger(CStateProperties *properties, CActionSet *actions, char* autoSavefile, int holdMemory) : CStateModifiersObject(properties), CEpisodeHistory(properties, actions)
{
	init();

	this->holdMemory = holdMemory;

	setAutoSaveFile(autoSavefile);	
	
	loadFileName[0] = '\0';
	loadModifiers = NULL;
}

CAgentLogger::CAgentLogger(char *loadFile, CStateProperties *properties, CActionSet *actions, std::list<CStateModifier *> *modifiers) :  CStateModifiersObject(properties), CEpisodeHistory(properties, actions)
{
	init();

	FILE* load = fopen(loadFile, "rb");
	
	sprintf(loadFileName, "%s", loadFile);
//	setmode(load, _O_BINARY);
	
	if (modifiers)
	{
		addStateModifiers(modifiers);
		loadModifiers = new std::list<CStateModifier *>(*modifiers);
	}
	else
	{
		loadModifiers = NULL;
	}

	this->holdMemory = -1;

	
	
	

	loadBIN(load, loadModifiers);

	fclose(load);
}

CAgentLogger::CAgentLogger(CStateProperties *properties, CActionSet *actions) : CStateModifiersObject(properties), CEpisodeHistory(properties, actions)
{
	init();
}

void CAgentLogger::init()
{
	file = NULL;
	episodes = new std::list<CEpisode *>();

	currentEpisode = new CEpisode(getStateProperties(), getActions());

    	holdMemory = -1;
    
   	loadFileName[0] = '\0';

	loadModifiers = NULL;
}


void CAgentLogger::setAutoSaveFile(char *filename)
{
	if (file != NULL)
	{
		fclose(file);
	}

	if (filename != NULL)
	{
		assert(strlen(filename) > 0);
		strcpy(this->filename, filename);
		file = fopen(filename,"ab");
//		setmode(file, _O_BINARY);
	}
	else file = NULL;
}

CAgentLogger::~CAgentLogger()
{
    if (file != NULL) 
	{
		fclose(file);
	}
	for (std::list<CEpisode *>::iterator it = episodes->begin(); it != episodes->end(); it++)
	{
		delete *it;
	}
	if (currentEpisode != NULL)
	{
		delete currentEpisode;
	}

	delete episodes;
	delete loadModifiers;
}

void CAgentLogger::resetData()
{
	for (std::list<CEpisode *>::iterator it = episodes->begin(); it != episodes->end(); it++)
	{
		delete *it;
	}
	if (currentEpisode != NULL)
	{
		delete currentEpisode;
		currentEpisode = NULL;
	}
	episodes->clear();
	
	if (strlen(loadFileName) > 0)
	{
		FILE* load = fopen(loadFileName, "rb");
	
		loadBIN(load, loadModifiers);

		fclose(load);
	}
}

void CAgentLogger::setLoadDataFile(char *loadData, std::list<CStateModifier *> *modifier)
{
	sprintf(loadFileName, "%s", loadData);

	if (modifier != NULL)
	{
		if (loadModifiers)
		{
			delete loadModifiers;
		}
		loadModifiers = new std::list<CStateModifier *>(*modifier);
	}
}

void CAgentLogger::nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState) 
{
	currentEpisode->nextStep(oldState, action, nextState);   
}

void CAgentLogger::newEpisode()
{
	if (currentEpisode != NULL)
	{
		if (currentEpisode->getNumSteps() > 0)
		{
			if (file != NULL)
			{
				currentEpisode->saveBIN(file);
				fflush(file);
			}	
			episodes->push_back(currentEpisode);
			currentEpisode = new CEpisode(getStateProperties(), getActions(), getStateModifiers());
		}
	}
	else
	{
		currentEpisode = new CEpisode(getStateProperties(), getActions(), getStateModifiers());
	}
	
	if (holdMemory > 0 && (int)episodes->size() > holdMemory)
	{
		CEpisode *firstEpisode = *episodes->begin();
		episodes->pop_front();
		delete firstEpisode;
	}
}

void CAgentLogger::saveBIN(FILE *stream)
{
	std::list<CEpisode *>::iterator it;

	for (it = episodes->begin(); it != episodes->end(); it ++)
	{
		(*it)->saveBIN(stream);
	}
}

void CAgentLogger::saveData(FILE *stream)
{
	std::list<CEpisode *>::iterator it;

//	fprintf(stream, "Episodes: %d\n", episodes->size());
	int i = 0;
	for (i = 0,it = episodes->begin(); it != episodes->end(); it ++, i++)
	{
		fprintf(stream, "\nEpisode: %d\n", i);
		(*it)->saveData(stream);
	}
}

void CAgentLogger::loadBIN(FILE *stream, std::list<CStateModifier *> *modifiers, int numEpisodes)
{
	int nEpisodes = numEpisodes;

	if (holdMemory >= 0 && holdMemory < numEpisodes)
	{
		holdMemory = numEpisodes;
	}
	
	std::list<CStateModifier *> *loadModifiers = modifiers;
	if (loadModifiers == NULL)
	{
		loadModifiers = getStateModifiers();
	}

	while (feof(stream) == 0 && (nEpisodes > 0 || nEpisodes < 0))
	{
		CEpisode  *tmp = new CEpisode(CEpisodeHistory::getStateProperties(), getActions(), loadModifiers);
		tmp->loadBIN(stream);

		if (tmp->getNumSteps() > 0)
		{
			episodes->push_back(tmp);
		}
		else
		{
			delete tmp;
		}

		nEpisodes --;
	}
	newEpisode();
}

void CAgentLogger::loadData(FILE *stream, int numEpisodes)
{
	int nEpisodes = numEpisodes;
	int buf;

	if (holdMemory >= 0 && holdMemory < numEpisodes)
	{
		holdMemory = numEpisodes;
	}

	while (feof(stream) == 0 && (nEpisodes > 0 || nEpisodes < 0))
	{
		CEpisode  *tmp = new CEpisode(getStateProperties(), getActions(), getStateModifiers());
		fscanf(stream, "\nEpisode: %d\n", &buf);
		tmp->loadData(stream);

		episodes->push_back(tmp);

		nEpisodes --;
	}
	
	newEpisode();
}

void CAgentLogger::loadData(FILE *stream)
{
	loadData(stream, -1);
}

int CAgentLogger::getNumEpisodes()
{
	return episodes->size();
}

CEpisode* CAgentLogger::getEpisode(int index)
{
	int i = 0;

	std::list<CEpisode *>::iterator it;

	assert(index < getNumEpisodes());

	for (it = episodes->begin(); it != episodes->end(), i < index; it++, i++)
	{
	
	}

	return *it;
}


void CAgentLogger::addStateModifier(CStateModifier *modifier)
{
	CStateModifiersObject::addStateModifier(modifier);

	if (currentEpisode)
	{
		currentEpisode->addStateModifier(modifier);
	}
}

void CAgentLogger::removeStateModifier(CStateModifier *modifier)
{
	CStateModifiersObject::removeStateModifier(modifier);

	if (currentEpisode)
	{
		currentEpisode->removeStateModifier(modifier);
	}
}

CEpisode* CAgentLogger::getCurrentEpisode()
{
	return currentEpisode;
}

void CAgentLogger::clearAutoSaveFile()
{
	if (file != NULL)
	{
		fclose(file);
		file = fopen(filename, "wb");
//		setmode(file, _O_BINARY);
	}
}

CEpisodeOutput::CEpisodeOutput(CStateProperties *featCalc, CRewardFunction *rewardFunction, CActionSet *actions, FILE *output) : CSemiMDPRewardListener(rewardFunction), CActionObject(actions), CStateObject(featCalc)
{
	this->stream = output;

	nSteps = 0;
	nEpisodes = 0;
}

CEpisodeOutput::~CEpisodeOutput()
{
}

void CEpisodeOutput::nextStep(CStateCollection *oldState, CAction *action, double reward,  CStateCollection *nextState)
{
	CActionData *actionData = action->getActionData();

	fprintf(stream, "Episode: %d, Step %d: ", nEpisodes, nSteps);
	fprintf(stream, "old State: ");
	oldState->getState(properties)->saveASCII(stream);
	fprintf(stream,"\nAction %d ", actions->getIndex(action));
	if (actionData != NULL)
	{
		actionData->saveASCII(stream);
	}
	fprintf(stream, " Reward %lf\n", reward);
	
	fprintf(stream, "newState: ");
	nextState->getState(properties)->saveASCII(stream);
	fprintf(stream, "\n");
	nSteps++;
	fflush(stream);
}

void CEpisodeOutput::newEpisode()
{
	fprintf(stream, "Start Episode %d\n", nEpisodes);
	nEpisodes++;
	nSteps = 0;
}

void CEpisodeOutput::intermediateStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	fprintf(stream, "Intermediate Step (%d)\n", nEpisodes);
	nextStep(oldState, action, reward, nextState);
}


CEpisodeMatlabOutput::CEpisodeMatlabOutput(CStateProperties *featCalc, CRewardFunction *rewardFunction, CActionSet *actions, FILE *output) : CSemiMDPRewardListener(rewardFunction), CActionObject(actions), CStateObject(featCalc)
{
	this->stream = output;

	nSteps = 0;
	nEpisodes = 0;
}

CEpisodeMatlabOutput::~CEpisodeMatlabOutput()
{
}

void CEpisodeMatlabOutput::nextStep(CStateCollection *oldState, CAction *action, double reward,  CStateCollection *nextState)
{
	CActionData *actionData = action->getActionData();

	fprintf(stream, "%d %d ", nEpisodes, nSteps);
	CState *state = oldState->getState(properties);
	unsigned int i = 0;
	for (i = 0; i < properties->getNumContinuousStates(); i++)
	{
		fprintf(stream, "%lf ", state->getContinuousState(i));
	}
	for (i = 0; i < properties->getNumDiscreteStates(); i++)
	{
		fprintf(stream, "%d ", (int)state->getDiscreteState(i));
	}
	
	state = nextState->getState(properties);
	for (i = 0; i < properties->getNumContinuousStates(); i++)
	{
		fprintf(stream, "%lf ", state->getContinuousState(i));
	}
	for (i = 0; i < properties->getNumDiscreteStates(); i++)
	{
		fprintf(stream, "%d ", (int)state->getDiscreteState(i));
	}
	
	fprintf(stream,"%d ", actions->getIndex(action));
	if (actionData != NULL)
	{
		CContinuousActionData *contData = dynamic_cast<CContinuousActionData *>(actionData);
		for (int j = 0; j < contData->nrows(); j++)
		{
			fprintf(stream, "%lf ", contData->element(j));
		}
	}
	
	fprintf(stream, "%lf ", reward);
		
	fprintf(stream, "\n");
	
	nSteps++;
	
	fflush(stream);
}

void CEpisodeMatlabOutput::newEpisode()
{
	nEpisodes++;
	nSteps = 0;
}

void CEpisodeMatlabOutput::setOutputFile(FILE *l_stream)
{
	stream = l_stream;
}

CEpisodeOutputStateChanged::CEpisodeOutputStateChanged(CStateProperties *featCalc, CRewardFunction *rewardFunction, CActionSet *actions, FILE *output) : CEpisodeOutput(featCalc, rewardFunction, actions, output)
{
}

void CEpisodeOutputStateChanged::nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *nextState)
{
	CState *old = oldState->getState(properties);
	CState *next = nextState->getState(properties);

	if (!old->equals(next))
	{
		CEpisodeOutput::nextStep(oldState, action, reward, nextState);
	}
	else
	{
		nSteps ++;
	}
}

CStateOutput::CStateOutput(CStateProperties *featCalc, FILE *output) : CStateObject(featCalc)
{
	this->stream = output;
}

CStateOutput::~CStateOutput()
{
}

void CStateOutput::nextStep(CStateCollection *, CAction *, CStateCollection *nextState)
{
//	if (isFirst)
//	{
//		oldState->getState(properties)->saveASCII(stream);		
//		isFirst = false;
//		printf("\n");
//	}
	unsigned int i = 0;
	CState *state = nextState->getState(properties);
	for (i = 0; i < properties->getNumContinuousStates(); i++)
	{
		fprintf(stream, "%lf ", state->getContinuousState(i));
	}
	for (i = 0; i < properties->getNumDiscreteStates(); i++)
	{
		fprintf(stream, "%d ", (int)state->getDiscreteState(i));
	}		
	fprintf(stream, "\n");
	fflush(stream);
}

CActionOutput::CActionOutput(CActionSet *actions, FILE *output) : CActionObject(actions)
{
	this->stream = output;
}

CActionOutput::~CActionOutput()
{
}

void CActionOutput::nextStep(CStateCollection *, CAction *action, CStateCollection *)
{
//	if (isFirst)
//	{
//		oldState->getState(properties)->saveASCII(stream);		
//		isFirst = false;
//		printf("\n");
//	}
	int index = actions->getIndex(action);
	fprintf(stream, "%d ", index);
	if (action->getActionData() != NULL)
	{
		action->getActionData()->saveASCII(stream);		
	}
	fprintf(stream, "\n");	
}

