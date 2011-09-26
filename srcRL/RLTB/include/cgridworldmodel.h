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

#ifndef CGRIDWORDMODEL_H
#define CGRIDWORDMODEL_H

#ifdef WIN32
#include <windows.h>
#endif // WIN32

#include "ctransitionfunction.h"
#include "crewardfunction.h"
#include "cagentcontroller.h"
#include "cdiscretizer.h"
#include "caction.h"

#include <math.h>
#include <vector>
#include <map>
#include <set>

#define GRIDWORLDACTION 16

class CGridWorld
{
protected:
	int size_x, size_y;
	std::set<char>* start_values;
	std::set<char>* target_values;
	std::set<char>* prohibited_values;

	std::vector<char *>* grid;

	void allocGrid();
	void deallocGrid();

	bool is_allocated;

public:	
	CGridWorld(char* filename);
	CGridWorld(unsigned int size_x, unsigned int size_y);
	virtual ~CGridWorld();

	void load(char* filename);
	virtual void load(FILE *stream);

	void save(char* filename);
	virtual void save(FILE *stream);

	virtual void initGrid();
	virtual bool isValid();

    virtual void setGridValue(unsigned int pos_x, unsigned int pos_y, char value);
    char getGridValue(unsigned int pos_x, unsigned int pos_y);

	virtual void addStartValue(char value);
	virtual void removeStartValue(char value);
	std::set<char> *getStartValues();

	virtual void addTargetValue(char value);
	virtual void removeTargetValue(char value);
	std::set<char> *getTargetValues();

	virtual void addProhibitedValue(char value);
	virtual void removeProhibitedValue(char value);
	std::set<char> *getProhibitedValues();

	void setSize(unsigned int size_x, unsigned int size_y);

	unsigned int getSizeX();
	unsigned int getSizeY();

	std::set<char> *getUsedValues();
};


class CGridWorldModel : public CGridWorld, public CTransitionFunction, public CRewardFunction
{
protected:
	unsigned int max_bounces;


	std::vector<std::pair<int, int>* >* start_points;
	std::map<char, double> *rewards;

	double reward_standard;
	double reward_success;
	double reward_bounce;

	bool is_parsed;
	virtual void parseGrid();

public:	
	CGridWorldModel(unsigned int size_x, unsigned int size_y, unsigned int max_bounces);	
	CGridWorldModel(char* filename, unsigned int max_bounces);	
	virtual ~CGridWorldModel();

	void setMaxBounces(unsigned int value);
	unsigned int getMaxBounces();

	/*void setActualBounces(unsigned int value);
	unsigned int getActualBounces();

	void setPosX(unsigned int value);
	int getPosX();

	void setPosY(unsigned int value);
	int getPosY();*/

    void setRewardStandard(double value);
	void setRewardSuccess(double value);
	void setRewardBounce(double value);

	void setRewardForSymbol(char symbol, double reward);
	double getRewardForSymbol(char symbol);

	double getRewardStandard();
	double getRewardSuccess();
	double getRewardBounce();

	virtual void load(FILE *stream);
	virtual void initGrid();
    virtual void setGridValue(unsigned int pos_x, unsigned int pos_y, char value);
	virtual void addStartValue(char value);
	virtual void removeStartValue(char value);

	virtual void transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *data = NULL);

	virtual bool isResetState(CState *state);
	virtual bool isFailedState(CState *state);

	virtual void getResetState(CState *resetState);

	virtual double getReward(CStateCollection *oldState, CAction *action, CStateCollection *newState);

};


class CLocal4GridWorldState : public CStateModifier
{
protected:
	CGridWorld* grid_world;
public:
	CLocal4GridWorldState(CGridWorld *grid_world);	
	virtual ~CLocal4GridWorldState();

	virtual void getModifiedState(CStateCollection *originalState, CState *modifiedState);
};


class CLocal4XGridWorldState : public CStateModifier
{
protected:
	CGridWorld* grid_world;
public:
	CLocal4XGridWorldState(CGridWorld *grid_world);	
	virtual ~CLocal4XGridWorldState();

	virtual void getModifiedState(CStateCollection *originalState, CState *modifiedState);
};


class CLocal8GridWorldState : public CStateModifier
{
protected:
	CGridWorld* grid_world;
public:
	CLocal8GridWorldState(CGridWorld *grid_world);	
	virtual ~CLocal8GridWorldState();

	virtual void getModifiedState(CStateCollection *originalState, CState *modifiedState);
};


class CGlobalGridWorldDiscreteState : public CAbstractStateDiscretizer
{
protected:
	unsigned int size_x, size_y;

public:
	CGlobalGridWorldDiscreteState(unsigned int size_x, unsigned int size_y);
	virtual ~CGlobalGridWorldDiscreteState() {};

	virtual unsigned int getDiscreteStateNumber(CStateCollection *state);		
};


class CLocalGridWorldDiscreteState : public CAbstractStateDiscretizer
{
protected:
	CStateModifier* orig_state;
	std::map<char, short>* valuemap;

public:
	CLocalGridWorldDiscreteState(CStateModifier* orig_state, unsigned int neigbourhood, std::set<char> *possible_values);
	virtual ~CLocalGridWorldDiscreteState();

	virtual unsigned int getDiscreteStateNumber(CStateCollection *state);		
};


class CSmallLocalGridWorldDiscreteState : public CAbstractStateDiscretizer
{
protected:
	CStateModifier* orig_state;
	CGridWorld *gridworld;

public:
	CSmallLocalGridWorldDiscreteState(CStateModifier* orig_state, unsigned int neigbourhood, CGridWorld *gridworld);
	virtual ~CSmallLocalGridWorldDiscreteState();

	virtual unsigned int getDiscreteStateNumber(CStateCollection *state);		
};


class CGridWorldAction : public CPrimitiveAction
{
protected:
	int x_move, y_move;

public:
	CGridWorldAction(int x_move, int y_move);

	int getXMove();
    int getYMove();
};


class CGridWorldController : public CAgentStatisticController, public CSemiMDPListener
{
	struct GridControllerRecord
	{
		CGridWorldAction* action;
		int pos_x;
		int pos_y;
		double factor;
		double distance;
	};

protected:
	CGridWorld *gridworld;
	std::vector<GridControllerRecord> *record;
	std::set<std::pair<unsigned int, unsigned int>*>* target_points;
	int lastXMove, lastYMove;

public:
	CGridWorldController(CGridWorld *gridworld, CActionSet *actions);
	virtual ~CGridWorldController();

	void init();

	virtual CAction* getNextAction(CStateCollection *state, CActionStatistics *stat);

	virtual void  nextStep(CStateCollection *, CAction *, CStateCollection *) {}; 
   
	virtual void  newEpisode();
};


#ifdef WIN32

class CGridWorldVisualizer : public CSemiMDPListener
{
protected:
	CGridWorldModel *gridworld;
	bool flgDisplay;
	bool flgTranspose;
	HANDLE console;
	short xpos, ypos, xoffset, yoffset;

public:
    CGridWorldVisualizer(CGridWorldModel *gridworld);
	virtual ~CGridWorldVisualizer();

	virtual void  nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState); 
   
	virtual void  intermediateStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState) {};

	virtual void  newEpisode();

	bool getDisplay();

	void setDisplay(bool flgDisplay);
};

#endif // WIN32


class CRaceTrackDiscreteState : public CAbstractStateDiscretizer
{
protected:
	CStateModifier* orig_state;
	CGridWorld *gridworld;

public:
	CRaceTrackDiscreteState(CStateModifier* orig_state, unsigned int neigbourhood, CGridWorld *gridworld);
	virtual ~CRaceTrackDiscreteState();

	virtual unsigned int getDiscreteStateNumber(CStateCollection *state);		
};


class CRaceTrack
{
public:
	static void generateRaceTrack(CGridWorld *gridworld, unsigned int width = 40, unsigned int length = 200, unsigned int h_max = 5, unsigned int dy_min = 1, unsigned int dy_max = 8);
};


#endif // CGRIDWORDMODEL_H

