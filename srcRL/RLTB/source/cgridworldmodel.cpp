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
#include "cgridworldmodel.h"

#include "cstate.h"
#include "cstateproperties.h"
#include "cstatecollection.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <stdexcept>



CGridWorld::CGridWorld(char* filename)
{
	start_values = new std::set<char>();
	target_values = new std::set<char>();
	prohibited_values = new std::set<char>();
	grid = new std::vector<char *>();
	size_x = 0;
	size_y = 0;
	load(filename);
}

CGridWorld::CGridWorld(unsigned int size_x, unsigned int size_y)
{
	start_values = new std::set<char>();
	target_values = new std::set<char>();
	prohibited_values = new std::set<char>();
	grid = new std::vector<char *>();
	this->size_x = size_x;
	this->size_y = size_y;
	initGrid();
}

CGridWorld::~CGridWorld()
{
	deallocGrid();
	start_values->clear();
	delete start_values;
	target_values->clear();
	delete target_values;
	prohibited_values->clear();
	delete prohibited_values;
	delete grid;
}

void CGridWorld::allocGrid()
{
	deallocGrid();
    for (int i = 0; i < size_y; i++)
	{
		grid->push_back(new char[size_x]);
	}
	is_allocated = true;
}

void CGridWorld::deallocGrid()
{
    if (is_allocated)
	{
		for (int i = 0; i < size_y; i++)
		{
			delete (*grid)[i];
		}
		grid->clear();
		is_allocated = false;
	}
}

void CGridWorld::initGrid()
{
	if (!is_allocated && size_x > 0 && size_y > 0)
	{
		allocGrid();
	}
}

bool CGridWorld::isValid()
{
	return is_allocated;
}

void CGridWorld::load(char* filename)
{
    FILE* stream = fopen(filename, "r");
	if (!stream) 
	{
		throw new std::runtime_error("Gridworld: Datei konnte nicht gefunden werden!");
	}
	load(stream);
	fclose(stream);
}

void CGridWorld::load(FILE *stream)
{
    int xsize = 0, ysize = 0, i = 0;
	bool flg_grid_world = false;
	char buffer[1024];
	std::string text;
	std::set<char>* tmpset;
	char temp;

    if (fgets(buffer, 1024, stream) == NULL)
	{
		throw new std::runtime_error("Gridworld: Datei enthaelt keine gueltige Gridworld! (unerwartetes Dateiende)");
	}
    
	while (!((buffer[0] <= 57 && buffer[0] >= 48) || (buffer[0] <= 70 && buffer[0] >= 65))) // != 1..9,A..F
	{
		if (!(buffer[0] == '#' || buffer[0] == 13 || buffer[0] == 10)) // ignore # and enter
		{
			text = buffer;
			if(text.find("Gridworld") == 0)
			{
				flg_grid_world = true;
			}
			else if (text.find("Size: ") == 0)
			{
				if (sscanf(buffer, "Size: %dx%d", &xsize, &ysize) != 2)
					throw new std::runtime_error("Gridworld: Datei enthaelt keine gueltigen Gridworldgroessenangaben!");
				this->size_x = xsize;
				this->size_y = ysize;
			}
			else
			{
				if (text.find("StartValues: ") == 0)
				{
					tmpset = start_values;
					i = strlen("StartValues:");
				}
				else if (text.find("TargetValues: ") == 0)
				{
					tmpset = target_values;
					i = strlen("TargetValues:");
				}
				else if (text.find("ProhibitedValues: ") == 0)
				{
					tmpset = prohibited_values;
					i = strlen("ProhibitedValues:");
				}
				else 
				{
					throw new std::runtime_error("Gridworld: Datei enthaelt keine gueltige Gridworld! (ungueltiges Token)");
				}

				while (i > 0)
				{
					if (sscanf(text.substr(i + 1).c_str(), "%c", &temp) != 1)
					{
						throw new std::runtime_error("Gridworld: Datei enthaelt keine gueltige Gridworld! (ungueltiger Wert)");
					}
					tmpset->insert((char)temp);
					i = text.find(",", i + 1);
				}
			}
		}

		if (fgets(buffer, 1024, stream) == NULL)
			throw new std::runtime_error("Gridworld: Datei enthaelt keine gueltige Gridworld! (unerwartetes Dateiende)");
	}

    if (!flg_grid_world)
		throw new std::runtime_error("Gridworld: Datei enthaelt keine gueltige Gridworld!");

	i = 0;
	if (size_x == 0) size_x = strlen(buffer) - 1;
		
	do 
	{
		if (strlen(buffer) < (unsigned int)(size_x))
			throw new std::runtime_error("Gridworld: Datei enthaelt keine gueltige Gridworld! (size_x != Gridgroesse)");
		grid->push_back(new char[size_x]);
		for (int j = size_x - 1; j >= 0; j--)
		{
			/*if (sscanf (&buffer[j], "%X", &temp) != 1)
				throw new std::runtime_error("Gridworld: Datei enthaelt keine gueltige Gridworld! (ungueltiger Gridwert)");*/
			(*grid)[i][j] = (char)buffer[j];
			buffer[j] = 0;
		}
		i++;
	}
	while (fgets(buffer, 1024, stream) != NULL);

	if (size_y == 0)
		size_y = i;
	if (i != size_y)
		throw new std::runtime_error("Gridworld: Datei enthaelt keine gueltige Gridworld! (size_y != Gridgroesse)");

	is_allocated = true;
}

void CGridWorld::save(char* filename)
{
    FILE* stream = fopen(filename, "w");
	if (!stream)
		throw new std::runtime_error("Gridworld: Datei konnte nicht erstellt werden!");
	save(stream);
	fclose(stream);
}

void CGridWorld::save(FILE *stream)
{
    fprintf(stream, "Gridworld\n");
	fprintf(stream, "Size: %dx%d\n", size_x, size_y);

	if (!start_values->empty())
	{
		std::set<char>::iterator it = start_values->begin();
		fprintf(stream, "StartValues: %1X", *(it++));
		while (it != start_values->end())
		{
			fprintf(stream, ", %1X", *(it++));
		}
		fprintf(stream, "\n");
	}

	if (!target_values->empty())
	{
		std::set<char>::iterator it = target_values->begin();
		fprintf(stream, "TargetValues: %1X", *(it++));
		while (it != target_values->end())
		{
			fprintf(stream, ", %1X", *(it++));
		}
		fprintf(stream, "\n");
	}

	if (!prohibited_values->empty())
	{
		std::set<char>::iterator it = prohibited_values->begin();
		fprintf(stream, "ProhibitedValues: %1X", *(it++));
		while (it != prohibited_values->end())
		{
			fprintf(stream, ", %1X", *(it++));
		}
		fprintf(stream, "\n");
	}

	fprintf(stream, "\n");
    for (int i = 0; i < size_y; i++)
	{
        for (int j = 0; j < size_x; j++)
		{
            fprintf(stream, "%1X", (*grid)[i][j]);
        }
        fprintf(stream, "\n");			
    }
}

void CGridWorld::setGridValue(unsigned int pos_x, unsigned int pos_y, char value)
{
	if (is_allocated && ((int)pos_x < size_x) && ((int)pos_y < size_y))
	{
		(*grid)[pos_y][pos_x] = value;
	}
}

char CGridWorld::getGridValue(unsigned int pos_x, unsigned int pos_y)
{
	if (is_allocated && ((int)pos_x < size_x) && ((int)pos_y < size_y))
	{
		return (*grid)[pos_y][pos_x];
	}
	else throw new std::invalid_argument("Gridworld_getGridValue: ungueltiger Parameter oder Grid nicht initialisiert)");
}

void CGridWorld::addStartValue(char value)
{
	start_values->insert(value);
}

void CGridWorld::removeStartValue(char value)
{
	start_values->erase(value);
}

std::set<char> *CGridWorld::getStartValues()
{
	return start_values;
}

void CGridWorld::addTargetValue(char value)
{
	target_values->insert(value);
}

void CGridWorld::removeTargetValue(char value)
{
	target_values->erase(value);
}

std::set<char> *CGridWorld::getTargetValues()
{
	return target_values;
}

void CGridWorld::addProhibitedValue(char value)
{
	prohibited_values->insert(value);
}

void CGridWorld::removeProhibitedValue(char value)
{
	prohibited_values->erase(value);
}

std::set<char> *CGridWorld::getProhibitedValues()
{
	return prohibited_values;
}

void CGridWorld::setSize(unsigned int size_x, unsigned int size_y)
{
	this->deallocGrid();
	this->size_x = size_x;
	this->size_y = size_y;
}

unsigned int CGridWorld::getSizeX()
{
	return this->size_x;
}

unsigned int CGridWorld::getSizeY()
{
	return this->size_y;
}

std::set<char> *CGridWorld::getUsedValues()
{
	static std::set<char> *values = new std::set<char>();
	values->clear();
	char tmp;
	for (int j = 0; j < size_y; j++)
	{
		for (int i = 0; i < size_x; i++)
		{
			tmp = (*grid)[j][i];
			values->insert((*grid)[j][i]);
		}
	}
	return values;
}


CGridWorldModel::CGridWorldModel(char* filename, unsigned int max_bounces) : CGridWorld(filename), CTransitionFunction(new CStateProperties(0,3), new CActionSet()) {
	this->max_bounces = max_bounces;
	this->reward_standard = -1.0;
	this->reward_bounce = -10;
	this->reward_success = 100.0;
	this->is_parsed = false;
	this->start_points = new std::vector<std::pair<int, int>* >();
	this->rewards = new std::map<char, double>();

	properties->setDiscreteStateSize(0, getSizeX());
	properties->setDiscreteStateSize(1, getSizeY());

	properties->setDiscreteStateSize(2, max_bounces+2);
}

CGridWorldModel::CGridWorldModel(unsigned int size_x, unsigned int size_y, unsigned int max_bounces) : CGridWorld(size_x, size_y), CTransitionFunction(new CStateProperties(0,3), new CActionSet()) {
	this->max_bounces = max_bounces;
	this->reward_standard = -1.0;
	this->reward_bounce = -10;
	this->reward_success = 100.0;
	this->is_parsed = false;
	this->start_points = new std::vector<std::pair<int, int>* >();
	this->rewards = new std::map<char, double>();

	properties->setDiscreteStateSize(0, getSizeX());
	properties->setDiscreteStateSize(1, getSizeY());
	properties->setDiscreteStateSize(2, max_bounces+2);
}

CGridWorldModel::~CGridWorldModel()
{
    for(unsigned int j = 0; j < start_points->size(); j++)
	{
		delete (*start_points)[j];
	}
	delete start_points;

	delete properties;
	delete actions;
	delete rewards;
}

void CGridWorldModel::setMaxBounces(unsigned int value)
{
	this->max_bounces = value;
	properties->setDiscreteStateSize(2, max_bounces+2);
}

unsigned int CGridWorldModel::getMaxBounces()
{
	return max_bounces;
}
/*
void CGridWorldModel::setActualBounces(unsigned int value)
{
	this->actual_bounces = value;
}

unsigned int CGridWorldModel::getActualBounces()
{
	return actual_bounces;
}

void CGridWorldModel::setPosX(unsigned int value)
{
	if (value >= (unsigned int)size_x)
		throw new std::invalid_argument("GridWorldModel_setPosX: ungueltiger Parameter!");
	pos_x = value;
}

int CGridWorldModel::getPosX()
{
	return pos_x;
}

void CGridWorldModel::setPosY(unsigned int value)
{
	if (value >= (unsigned int)size_y)
		throw new std::invalid_argument("GridWorldModel_setPosY: ungueltiger Parameter!");
	pos_y = value;
}

int CGridWorldModel::getPosY()
{
	return pos_y;
}
*/
void CGridWorldModel::setRewardStandard(double value)
{
	this->reward_standard = value;
	
}

double CGridWorldModel::getRewardStandard()
{
	return this->reward_standard;
}

void CGridWorldModel::setRewardSuccess(double value)
{
	this->reward_success = value;

}

double CGridWorldModel::getRewardSuccess()
{
	return this->reward_success;
}

void CGridWorldModel::setRewardBounce(double value)
{
	reward_bounce = value;
}


double CGridWorldModel::getRewardBounce()
{
	return reward_bounce;
}

void CGridWorldModel::setRewardForSymbol(char symbol, double reward)
{
	(*rewards)[symbol] = reward;
}

double CGridWorldModel::getRewardForSymbol(char symbol)
{
	std::map<char, double>::iterator it = rewards->find(symbol);
	double rew = 0.0;

	if (it != rewards->end())
	{
		rew = (*it).second;
	}
	else
	{
		if (target_values->find(symbol) != target_values->end())
		{
			rew = reward_success;
		}
		else
		{
			rew = reward_standard;
		}
	}
	return rew;
}



void CGridWorldModel::parseGrid()
{
	if (isValid() && !is_parsed)
	{
		for(unsigned int h = 0; h < start_points->size(); h++)
		{
			delete (*start_points)[h];
		}
		start_points->clear();
	    for (int j = 0; j < size_y; j++)
		{
			for (int i = 0; i < size_x; i++)
			{
				if(start_values->find(getGridValue(i, j)) != start_values->end())
					start_points->push_back(new std::pair<int, int>(j, i));
			}
		}
	}
	is_parsed = true;
}

void CGridWorldModel::load(FILE *stream)
{
	CGridWorld::load(stream);
	is_parsed = false;
}

void CGridWorldModel::initGrid()
{
	CGridWorld::initGrid();
	is_parsed = false;
}

void CGridWorldModel::setGridValue(unsigned int pos_x, unsigned int pos_y, char value)
{
	CGridWorld::setGridValue(pos_x, pos_y, value);
	is_parsed = false;
}

void CGridWorldModel::addStartValue(char value)
{
	CGridWorld::addStartValue(value);
	is_parsed = false;
}

void CGridWorldModel::removeStartValue(char value)
{
	CGridWorld::removeStartValue(value);
	is_parsed = false;
}

void CGridWorldModel::transitionFunction(CState *oldstate, CAction *action, CState *newState, CActionData *)
{
	int pos_x = oldstate->getDiscreteState(0);
	int pos_y = oldstate->getDiscreteState(1);
	int actual_bounces = oldstate->getDiscreteState(2);

	CGridWorldAction* gridAction = dynamic_cast<CGridWorldAction*>(action);

	int tmp_pos_x = pos_x + gridAction->getXMove();
	int tmp_pos_y = pos_y + gridAction->getYMove();

	if (tmp_pos_x < 0 || tmp_pos_x >= size_x || tmp_pos_y < 0 || tmp_pos_y >= size_y)
	{
		actual_bounces++;
	}
	else if (prohibited_values->find(getGridValue(tmp_pos_x, tmp_pos_y)) != prohibited_values->end())
	{
		actual_bounces++;
	}
	else
	{
		pos_x = tmp_pos_x;
		pos_y = tmp_pos_y;
	}

	newState->setDiscreteState(0, pos_x);
	newState->setDiscreteState(1, pos_y);
	newState->setDiscreteState(2,actual_bounces);
}

bool CGridWorldModel::isResetState(CState *state)
{
	return ((unsigned int) state->getDiscreteState(2) > this->max_bounces) || (target_values->find(getGridValue(state->getDiscreteState(0), state->getDiscreteState(1))) != target_values->end());
}

bool CGridWorldModel::isFailedState(CState *state)
{
	return ((unsigned int) state->getDiscreteState(2) > this->max_bounces);
}

void CGridWorldModel::getResetState(CState *resetState)
{
	if (!is_parsed)
		parseGrid();
	if (start_points->size() > 0)
	{
		int i = rand() % start_points->size();
		resetState->setDiscreteState(0, (*start_points)[i]->second);
		resetState->setDiscreteState(1, (*start_points)[i]->first);
	}
	else
	{
		resetState->setDiscreteState(0, 0);
		resetState->setDiscreteState(1, 0);
	}
	resetState->setDiscreteState(2, 0);
}

double CGridWorldModel::getReward(CStateCollection *oldState, CAction *, CStateCollection *newState) {
	double rew = 0.0;
	if (newState->getState()->getDiscreteState(2) > oldState->getState()->getDiscreteState(2))
	{
		rew = reward_bounce;
	}
	else
	{
		int x = newState->getState()->getDiscreteState(0);
		int y = newState->getState()->getDiscreteState(1);

		rew = getRewardForSymbol(getGridValue(x, y));
	}
    return rew;
} 

CLocal4GridWorldState::CLocal4GridWorldState(CGridWorld* grid_world) : CStateModifier(0, 4)
{
	this->grid_world = grid_world;
	for (int i = 0; i < 4; i++)
	{
		setDiscreteStateSize(i, grid_world->getUsedValues()->size());
	}
}

CLocal4GridWorldState::~CLocal4GridWorldState()
{
}

void CLocal4GridWorldState::getModifiedState(CStateCollection *originalState, CState *state)
{
	int pos_x = originalState->getState()->getDiscreteState(0);
	int pos_y = originalState->getState()->getDiscreteState(1);
	
	if (pos_y > 0)
		state->setDiscreteState(0, grid_world->getGridValue(pos_x, pos_y - 1));
	else
		state->setDiscreteState(0, grid_world->getGridValue(pos_x, pos_y));

	if (pos_x < (int)grid_world->getSizeX() - 1)
		state->setDiscreteState(1, grid_world->getGridValue(pos_x + 1, pos_y));
	else
		state->setDiscreteState(1, grid_world->getGridValue(pos_x, pos_y));

	if (pos_y < (int)grid_world->getSizeY() - 1)
		state->setDiscreteState(2, grid_world->getGridValue(pos_x, pos_y + 1));
	else
		state->setDiscreteState(2, grid_world->getGridValue(pos_x, pos_y));

	if (pos_x > 0)
		state->setDiscreteState(3, grid_world->getGridValue(pos_x - 1, pos_y));
	else
		state->setDiscreteState(3, grid_world->getGridValue(pos_x, pos_y));
} 


CLocal4XGridWorldState::CLocal4XGridWorldState(CGridWorld* grid_world) : CStateModifier(0, 4)
{
	this->grid_world = grid_world;
	for (int i = 0; i < 4; i++)
	{
		setDiscreteStateSize(i, grid_world->getUsedValues()->size());
	}
}

CLocal4XGridWorldState::~CLocal4XGridWorldState()
{
}

void CLocal4XGridWorldState::getModifiedState(CStateCollection *originalState, CState *state)
{
	int pos_x = originalState->getState()->getDiscreteState(0);
	int pos_y = originalState->getState()->getDiscreteState(1);
	
	if ((pos_y > 0) && (pos_x > 0))
		state->setDiscreteState(0, grid_world->getGridValue(pos_x - 1, pos_y - 1));
	else
		state->setDiscreteState(0, grid_world->getGridValue(pos_x, pos_y));

	if ((pos_y > 0) && (pos_x < (int)grid_world->getSizeX() - 1))
		state->setDiscreteState(1, grid_world->getGridValue(pos_x + 1, pos_y - 1));
	else
		state->setDiscreteState(1, grid_world->getGridValue(pos_x, pos_y));

	if ((pos_y < (int)grid_world->getSizeY() - 1) && (pos_x < (int)grid_world->getSizeX() - 1))
		state->setDiscreteState(2, grid_world->getGridValue(pos_x + 1, pos_y + 1));
	else
		state->setDiscreteState(2, grid_world->getGridValue(pos_x, pos_y));

	if ((pos_y < (int)grid_world->getSizeY() - 1) && (pos_x > 0))
		state->setDiscreteState(3, grid_world->getGridValue(pos_x - 1, pos_y + 1));
	else
		state->setDiscreteState(3, grid_world->getGridValue(pos_x, pos_y));
} 


CLocal8GridWorldState::CLocal8GridWorldState(CGridWorld* grid_world) : CStateModifier(0, 8)
{
	this->grid_world = grid_world;
	for (int i = 0; i < 8; i++)
	{
		setDiscreteStateSize(i, grid_world->getUsedValues()->size());
	}
}

CLocal8GridWorldState::~CLocal8GridWorldState()
{
}

void CLocal8GridWorldState::getModifiedState(CStateCollection *originalState, CState *state)
{
	int pos_x = originalState->getState()->getDiscreteState(0);
	int pos_y = originalState->getState()->getDiscreteState(1);

	if (pos_y > 0)
		state->setDiscreteState(0, grid_world->getGridValue(pos_x, pos_y - 1));
	else
		state->setDiscreteState(0, grid_world->getGridValue(pos_x, pos_y));

	if ((pos_y > 0) && (pos_x < (int)grid_world->getSizeX() - 1))
		state->setDiscreteState(1, grid_world->getGridValue(pos_x + 1, pos_y - 1));
	else
		state->setDiscreteState(1, grid_world->getGridValue(pos_x, pos_y));

	if (pos_x < (int)grid_world->getSizeX() - 1)
		state->setDiscreteState(2, grid_world->getGridValue(pos_x + 1, pos_y));
	else
		state->setDiscreteState(2, grid_world->getGridValue(pos_x, pos_y));

	if ((pos_y < (int)grid_world->getSizeY() - 1) && (pos_x < (int)grid_world->getSizeX() - 1))
		state->setDiscreteState(3, grid_world->getGridValue(pos_x + 1, pos_y + 1));
	else
		state->setDiscreteState(3, grid_world->getGridValue(pos_x, pos_y));

	if (pos_y < (int)grid_world->getSizeY() - 1)
		state->setDiscreteState(4, grid_world->getGridValue(pos_x, pos_y + 1));
	else
		state->setDiscreteState(4, grid_world->getGridValue(pos_x, pos_y));

	if ((pos_y < (int)grid_world->getSizeY() - 1) && (pos_x > 0))
		state->setDiscreteState(5, grid_world->getGridValue(pos_x - 1, pos_y + 1));
	else
		state->setDiscreteState(5, grid_world->getGridValue(pos_x, pos_y));

	if (pos_x > 0)
		state->setDiscreteState(6, grid_world->getGridValue(pos_x - 1, pos_y));
	else
		state->setDiscreteState(6, grid_world->getGridValue(pos_x, pos_y));

	if ((pos_y > 0) && (pos_x > 0))
		state->setDiscreteState(7, grid_world->getGridValue(pos_x - 1, pos_y - 1));
	else
		state->setDiscreteState(7, grid_world->getGridValue(pos_x, pos_y));
} 


CGlobalGridWorldDiscreteState::CGlobalGridWorldDiscreteState(unsigned int size_x, unsigned int size_y) : CAbstractStateDiscretizer(size_x * size_y + 1)
{
	this->size_x = size_x;
	this->size_y = size_y;
}

unsigned int CGlobalGridWorldDiscreteState::getDiscreteStateNumber(CStateCollection *state) {
    unsigned int discstate;
	int x = state->getState()->getDiscreteState(0);
	int y = state->getState()->getDiscreteState(1);

	if (x < 0 || (unsigned int)x >= size_x || y < 0 || (unsigned int)y >= size_y)
	{
		discstate = 0;
    }
	else
	{
		discstate = y * size_x + x + 1;
	}
	return discstate;
}


CLocalGridWorldDiscreteState::CLocalGridWorldDiscreteState(CStateModifier* orig_state, unsigned int neighbourhood, std::set<char> *possible_values) : CAbstractStateDiscretizer((int)pow((double) possible_values->size(), (double) neighbourhood))
{
	this->orig_state = orig_state;
	valuemap = new std::map<char, short>();
	std::set<char>::iterator it = possible_values->begin();
	for(short i = 0; it != possible_values->end(); i++, it++)
	{
		(*valuemap)[(*it)] = i;
	}
}

CLocalGridWorldDiscreteState::~CLocalGridWorldDiscreteState()
{
	valuemap->clear();
	delete valuemap;
}

unsigned int CLocalGridWorldDiscreteState::getDiscreteStateNumber(CStateCollection *state)
{
    CState *source_state = state->getState(orig_state);
    unsigned int discstate = 0;
	for (unsigned int i = 0; i < source_state->getNumDiscreteStates() - 1; i++)
	{
		discstate = discstate * valuemap->size() + (unsigned int)((*valuemap)[(char)source_state->getDiscreteState(i)]);
	}
	return discstate;
}


CSmallLocalGridWorldDiscreteState::CSmallLocalGridWorldDiscreteState(CStateModifier* orig_state, unsigned int neighbourhood, CGridWorld *gridworld) : CAbstractStateDiscretizer((unsigned int)pow((double) 3.0,(double) neighbourhood))
{
	this->orig_state = orig_state;
	this->gridworld = gridworld;
}

CSmallLocalGridWorldDiscreteState::~CSmallLocalGridWorldDiscreteState()
{
}

unsigned int CSmallLocalGridWorldDiscreteState::getDiscreteStateNumber(CStateCollection *state)
{
    CState *source_state = state->getState(orig_state);
    unsigned int discstate = 0;
	unsigned int temp;
	for (unsigned int i = 0; i < source_state->getNumDiscreteStates() - 1; i++)
	{
		if (gridworld->getTargetValues()->find((char)source_state->getDiscreteState(i)) != gridworld->getTargetValues()->end())
			temp = 2;
		else if (gridworld->getProhibitedValues()->find((char)source_state->getDiscreteState(i)) != gridworld->getProhibitedValues()->end())
			temp = 1;
		else
			temp = 0;
		discstate = discstate * 3 + temp;
	}
	return discstate;
}


CGridWorldAction::CGridWorldAction(int x_move, int y_move)
{
	addType(GRIDWORLDACTION);

    this->x_move = x_move;
	this->y_move = y_move;
}

int CGridWorldAction::getXMove()
{
    return this->x_move;
}

int CGridWorldAction::getYMove()
{
    return this->y_move;
}


#ifdef WIN32

CGridWorldController::CGridWorldController(CGridWorld *gridworld, CActionSet *actions) : CAgentStatisticController(actions)
{
    this->gridworld = gridworld;
	target_points = new std::set<std::pair<unsigned int, unsigned int>*>();
	record = new std::vector<GridControllerRecord>();
	record->resize(actions->size());
	lastXMove = 0;
	lastYMove = 0;
	init();
}

CGridWorldController::~CGridWorldController()
{
	record->clear();
	delete record;
	if (target_points->size() > 0)
	{
		std::set<std::pair<unsigned int, unsigned int> *>::iterator it = target_points->begin();
        for(;it != target_points->end(); it++)
		{
			delete *it;
		}
		target_points->clear();
	}
    delete target_points;
}

void CGridWorldController::init()
{
	std::pair<unsigned int, unsigned int> *target_point;
	if (target_points->size() > 0)
	{
		std::set<std::pair<unsigned int, unsigned int> *>::iterator it = target_points->begin();
        for(;it != target_points->end(); it++)
		{
			delete *it;
		}
		target_points->clear();
	}

	for (unsigned int j = 0; j < gridworld->getSizeY(); j++)
	{
		for (unsigned int i = 0; i < gridworld->getSizeX(); i++)
		{
			if (gridworld->getTargetValues()->find(gridworld->getGridValue(i ,j)) != gridworld->getTargetValues()->end())
			{
				target_point = new std::pair<unsigned int, unsigned int>(i, j);
				target_points->insert(target_point);
			}
		}
	}
}

void CGridWorldController::newEpisode()
{
	for (unsigned int i = 0; i < record->size(); i++)
	{
		(*record)[i].factor = 1.0;
	}
}

CAction* CGridWorldController::getNextAction(CStateCollection *state, CActionStatistics *stat)
{
	std::set<std::pair<unsigned int, unsigned int> *>::iterator itp;
	int x = state->getState()->getDiscreteState(0);
	int y = state->getState()->getDiscreteState(1);
	unsigned int bestind = 0;
	double dist;
	double maxdist = gridworld->getSizeX() * gridworld->getSizeX() + gridworld->getSizeY() * gridworld->getSizeY();
	CActionSet::iterator it = actions->begin();
	for (int i = 0; it != actions->end(); i++, it++)
	{
		CGridWorldAction *gridAction = dynamic_cast<CGridWorldAction*>(*it);
		(*record)[i].pos_x = x + gridAction->getXMove();
		(*record)[i].pos_y = y + gridAction->getYMove();
		(*record)[i].action = gridAction;
		dist = maxdist * 2.0;
		if (gridworld->getProhibitedValues()->find(gridworld->getGridValue((*record)[i].pos_x ,(*record)[i].pos_y)) == gridworld->getProhibitedValues()->end())
		{ // kein verbotener wert
			for (itp = target_points->begin(); itp != target_points->end(); itp++)
			{
		        dist = min(dist, ((*record)[i].pos_x - (*itp)->first) * ((*record)[i].pos_x - (*itp)->first) + ((*record)[i].pos_y - (*itp)->second) * ((*record)[i].pos_y - (*itp)->second));
			}
			dist *= (*record)[i].factor;
		}
		else
		{
			//(*record)[i].factor = 1.0;
		}
		if ((lastXMove == -gridAction->getXMove()) && (lastYMove == -gridAction->getYMove()))
		{ // wir wollen doch nicht den gleichen weg zurueck gehen
			(*record)[i].distance = max(maxdist - 0.5, dist);
		}
		else
		{
			(*record)[i].distance = dist;
		}
		if ((*record)[i].distance < (*record)[bestind].distance)
			bestind = i;
	}
	lastXMove = (*record)[bestind].action->getXMove();
	lastYMove = (*record)[bestind].action->getYMove();

	for (unsigned int j = 0; j < record->size(); j++)
	{
		if (j == bestind)
			(*record)[j].factor = max(0.7, (*record)[j].factor * 0.99);
		else
			(*record)[j].factor = min(1.3, (*record)[j].factor * 1.01);
	}

	if (stat != NULL)
	{
		stat->action = (*record)[bestind].action;
		stat->owner = this;
		stat->equal = 1;
		stat->probability = 0.5;
		stat->superior = 0;
		return stat->action;
	}
	else
	{
		return (*record)[bestind].action;
	}
}


CGridWorldVisualizer::CGridWorldVisualizer(CGridWorldModel *gridworld)
{
	this->gridworld = gridworld;
	this->console = GetStdHandle(STD_OUTPUT_HANDLE);
	this->flgDisplay = true;
	this->flgTranspose = false;
	this->xpos = 0;
    this->ypos = 0;
	this->xoffset = 1;
	this->yoffset = 3;
}

CGridWorldVisualizer::~CGridWorldVisualizer()
{
}

void CGridWorldVisualizer::nextStep(CStateCollection *oldStateCol, CAction *action, CStateCollection *nextStateCol)
{
	COORD coord;
    const WORD attribute = BACKGROUND_RED;
	DWORD dummy;

	CState *nextState = nextStateCol->getState();
	CState *oldState = oldStateCol->getState();

	if (flgTranspose)
	{
		xpos = (short)oldState->getDiscreteState(1);
		ypos = (short)oldState->getDiscreteState(0);
	}
	else
	{
		xpos = (short)oldState->getDiscreteState(0);
		ypos = (short)oldState->getDiscreteState(1);
	}

	if (flgDisplay)
	{
		coord.X = xpos + xoffset;
		coord.Y = ypos + yoffset;
		SetConsoleCursorPosition(console, coord);
		if (gridworld->getGridValue(xpos, ypos) != 0)
		{
			printf("%1X", gridworld->getGridValue(xpos, ypos));
		}
		else
		{
			printf(" ");
		}
		WriteConsoleOutputAttribute(console, &attribute, 1, coord, &dummy);

		if (flgTranspose)
		{
			xpos = (short)nextState->getDiscreteState(1);
			ypos = (short)nextState->getDiscreteState(0);
		}
		else
		{
			xpos = (short)nextState->getDiscreteState(0);
			ypos = (short)nextState->getDiscreteState(1);
		}
		coord.X = xpos + xoffset;
		coord.Y = ypos + yoffset;
		SetConsoleCursorPosition(console, coord);
		printf("#");
	}
}

void CGridWorldVisualizer::newEpisode()
{
	DWORD dummy;
	COORD coord;
	CONSOLE_SCREEN_BUFFER_INFO sInfo;
	GetConsoleScreenBufferInfo(console, &sInfo);
	coord.X = 0;
	coord.Y = 0;
	FillConsoleOutputCharacter(console, ' ', (1 + sInfo.srWindow.Right - sInfo.srWindow.Left) * (1 + sInfo.srWindow.Bottom - sInfo.srWindow.Top), coord, &dummy);
	
	if (flgDisplay)
	{
		flgTranspose = (gridworld->getSizeX() < gridworld->getSizeY());
		if (flgTranspose)
		{
			coord.X = xoffset;
			for (unsigned int h = 0; h < gridworld->getSizeX(); h++)
			{
				coord.Y = (short)h + yoffset;
				SetConsoleCursorPosition(console, coord);
				for (unsigned int j = 0; j < gridworld->getSizeY(); j++)
				{
					if (gridworld->getGridValue(h, j) != 0)
					{
						printf("%1X", gridworld->getGridValue(h, j));
					}
					else
					{
						printf(" ");
					}
				}
			}
			//xpos = (short)gridworld->getPosY();
			//ypos = (short)gridworld->getPosX();
		}
		else
		{
			coord.X = xoffset;
			for (unsigned int h = 0; h < gridworld->getSizeY(); h++)
			{
				coord.Y = (short)h + yoffset;
				SetConsoleCursorPosition(console, coord);
				for (unsigned int j = 0; j < gridworld->getSizeX(); j++)
				{
					if (gridworld->getGridValue(j, h) != 0)
					{
						printf("%1X", gridworld->getGridValue(j, h));
					}
					else
					{
						printf(" ");
					}
				}
			}
			//xpos = (short)gridworld->getPosX();
			//ypos = (short)gridworld->getPosY();
		}
		coord.X = xpos + xoffset;
		coord.Y = ypos + yoffset;
		//SetConsoleCursorPosition(console, coord);
		//printf("#");
	}
}

bool CGridWorldVisualizer::getDisplay()
{
	return flgDisplay;
}

void CGridWorldVisualizer::setDisplay(bool flgDisplay)
{
	this->flgDisplay = flgDisplay;
}

#endif // WIN32


void CRaceTrack::generateRaceTrack(CGridWorld *gridworld, unsigned int width, unsigned int length, unsigned int h_max, unsigned int dx_min, unsigned int dx_max)
{
	unsigned int i, j, x, dx, tmp1, tmp2, y1, y2, h;
	gridworld->setSize(length, width);
	gridworld->initGrid();
	gridworld->addProhibitedValue(1);
	gridworld->addProhibitedValue(2);
	gridworld->addStartValue(3);
	gridworld->addTargetValue(4);

	for (j = 1; j < width - 1; j++)
	{
		gridworld->setGridValue(0, j, 2);
		gridworld->setGridValue(1, j, 3);
		gridworld->setGridValue(length - 2, j, 4);
		gridworld->setGridValue(length - 1, j, 2);
		for (i = 2; i < length - 2; i++)
		{
			gridworld->setGridValue(i, j, 0);
		}

	}
	for (i = 0; i < length; i++)
	{
		gridworld->setGridValue(i, 0, 2);
		gridworld->setGridValue(i, width - 1, 2);
	}

	dx = (int)ceil((double)rand() / (double)RAND_MAX * (double)dx_max);
	x = 2 + dx;
	while (x < length - 2)
	{
		tmp1 = (int)ceil((double)rand() / (double)RAND_MAX * (double)(width - 1));
		tmp2 = (int)ceil((double)rand() / (double)RAND_MAX * (double)(width - 1));
		y1 = min(tmp1, tmp2);
		y2 = max(tmp1, tmp2);
		if (( y1 < 2) && (y2 > width - 3))
		{
			if (rand() < (RAND_MAX / 2))
			{
				y1 = 2;
				y2 = width - 2;
			}
			else
			{
				y1 = 1;
				y2 = width - 3;
			}
		}
		h = min(dx, (unsigned int)ceil((double)rand() / (double)RAND_MAX * (double)h_max)) - 1;
		for(i = y1; i <= y2; i++)
		{
			gridworld->setGridValue(x, i, 1);
		}
        for(i = x-h+1; i <= x; i++)
		{
			gridworld->setGridValue(i, y1, 1);
			gridworld->setGridValue(i, y2, 1);
		}
		dx = dx_min + (int)ceil((double)rand() / (double)RAND_MAX * (double)dx_max);
		x += dx;
	}
}


CRaceTrackDiscreteState::CRaceTrackDiscreteState(CStateModifier* orig_state, unsigned int neighbourhood, CGridWorld *gridworld) : CAbstractStateDiscretizer((unsigned int)pow((double) 2.0, (double) neighbourhood))
{
	this->orig_state = orig_state;
	this->gridworld = gridworld;
}

CRaceTrackDiscreteState::~CRaceTrackDiscreteState()
{
}

unsigned int CRaceTrackDiscreteState::getDiscreteStateNumber(CStateCollection *state)
{
    CState *source_state = state->getState(orig_state);
    unsigned int discstate = 0;
	unsigned int temp;
	for (unsigned int i = 0; i < source_state->getNumDiscreteStates() - 1; i++)
	{
        if (gridworld->getProhibitedValues()->find((char)source_state->getDiscreteState(i)) != gridworld->getProhibitedValues()->end())
			temp = 1;
		else
			temp = 0;
		discstate = discstate * 2 + temp;
	}
	return discstate;
}
