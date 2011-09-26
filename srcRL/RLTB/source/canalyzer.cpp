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
#include "canalyzer.h"
#include "cqfunction.h"
#include "cstatecollection.h"
#include "caction.h"
#include "cstateproperties.h"
#include "cstate.h"
#include "cvfunction.h"
#include "chistory.h"
#include "cepisodehistory.h"
#include "cinputdata.h"
#include "ckdtrees.h"
#include "cnearestneighbor.h"
#include "cevaluator.h"

#include <math.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>

CVFunctionAnalyzer::CVFunctionAnalyzer(CAbstractVFunction *vFunction, CStateProperties *modelState, std::list<CStateModifier *> *modifiers)
{
	this->vFunction = vFunction;
	this->modelStateProperties = modelState;

	stateCollection = new CStateCollectionImpl(modelState, modifiers);
}

CVFunctionAnalyzer::~CVFunctionAnalyzer()
{
	delete stateCollection;
}

void CVFunctionAnalyzer::save1DValues(FILE *stream, CState *initState, int dim1, int part1)
{
	double width = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);
	fprintf(stream, "VFunction Analyzer 1D-Table for Dimension %d\n", dim1);
	fprintf(stream, "InitialState: ");
	initState->saveASCII(stream);
	fprintf(stream, "\n");

	for (int i = 0; i <= part1; i ++)
	{
		double x = modelStateProperties->getMinValue(dim1) + i * width / part1;
		modelState->setContinuousState(dim1, x);
		stateCollection->newModelState();
		double value = vFunction->getValue(stateCollection);
		fprintf(stream, "%f %f\n", x, value);
	}
}

void CVFunctionAnalyzer::save1DMatlab(FILE *stream, CState *initState, int dim1, int part1)
{
	double width = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);

	for (int i = 0; i <= part1; i ++)
	{
		double x = modelStateProperties->getMinValue(dim1) + i * width / part1;
		modelState->setContinuousState(dim1, x);
		stateCollection->newModelState();
		double value = vFunction->getValue(stateCollection);
		fprintf(stream, "%f %f\n", x, value);
	}
}


void CVFunctionAnalyzer::saveDiscrete1DValues(FILE *stream, CState *initState, int dim1)
{
  int numDiscStates = modelStateProperties->getDiscreteStateSize(dim1);

  CState *modelState = stateCollection->getState(modelStateProperties);

  fprintf(stream, "VFunction Analyzer 1D-Table for Dimension %d\n", dim1);
  fprintf(stream, "InitialState: ");
  initState->saveASCII(stream);
  fprintf(stream, "\n");

  modelState->setState(initState);

  for (int i = 0; i < numDiscStates; i ++)
    {
      modelState->setDiscreteState(dim1, i);
      stateCollection->newModelState();
      double value = vFunction->getValue(stateCollection);
      fprintf(stream, "%d %f\n", i, value);
    }
}

void CVFunctionAnalyzer::saveDiscrete1DMatlab(FILE *stream, CState *initState, int dim1)
{
  	int numDiscStates = modelStateProperties->getDiscreteStateSize(dim1);

  	CState *modelState = stateCollection->getState(modelStateProperties);
  
  	modelState->setState(initState);

  	for (int i = 0; i < numDiscStates; i ++)
    	{
      		modelState->setDiscreteState(dim1, i);
      		stateCollection->newModelState();
      		double value = vFunction->getValue(stateCollection);
      		fprintf(stream, "%d %f\n", i, value);
    	}
}


void CVFunctionAnalyzer::save2DValues(FILE *stream, CState *initState, int dim1, int part1, int dim2, int part2)
{
	double width1 = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	double width2 = modelStateProperties->getMaxValue(dim2) - modelStateProperties->getMinValue(dim2);

	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);
	fprintf(stream, "VFunction Analyzer 2D-Table for Dimensions %d and %d\n", dim1, dim2);
	fprintf(stream, "InitialState: ");
	initState->saveASCII(stream);
	fprintf(stream, "\n");


	for (int i = 0; i <= part1; i ++)
	{

		double x = modelStateProperties->getMinValue(dim1) + i * width1 / part1;

		modelState->setContinuousState(dim1, x);

		for (int j = 0; j <= part2; j ++)
		{

			double y = modelStateProperties->getMinValue(dim2) + j * width2 / part2;

			modelState->setContinuousState(dim2, y);

			stateCollection->newModelState();

			double value = vFunction->getValue(stateCollection);

			fprintf(stream, "%f ", value);
		}
		fprintf(stream, "\n");
	}
}

void CVFunctionAnalyzer::save2DMatlab(FILE *stream, CState *initState, int dim1, int part1, int dim2, int part2)
{
	double width1 = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	double width2 = modelStateProperties->getMaxValue(dim2) - modelStateProperties->getMinValue(dim2);

	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);

	fprintf(stream, "VFunction Analyzer 2D-Table for Dimensions %d and %d\n", dim1, dim2);
	fprintf(stream, "InitialState: ");
	initState->saveASCII(stream);
	fprintf(stream, "\n");


	for (int i = 0; i <= part1; i ++)
	{
		double x = modelStateProperties->getMinValue(dim1) + i * width1 / part1;

		modelState->setContinuousState(dim1, x);

		for (int j = 0; j <= part2; j ++)
		{

			double y = modelStateProperties->getMinValue(dim2) + j * width2 / part2;

			modelState->setContinuousState(dim2, y);

			stateCollection->newModelState();

			double value = vFunction->getValue(stateCollection);

			fprintf(stream, "%f %f %f\n", x, y, value);
		}
	}
}

void CVFunctionAnalyzer::saveDiscrete2DValues(FILE *stream, CState *initState, int row, int col)
{
  int numDiscStates1 = modelStateProperties->getDiscreteStateSize(row);
  int numDiscStates2 = modelStateProperties->getDiscreteStateSize(col);

  CState *modelState = stateCollection->getState(modelStateProperties);

  modelState->setState(initState);

  for (int i = 0; i < numDiscStates1; i ++)
    {

      modelState->setDiscreteState(row, i);

      for (int j = 0; j < numDiscStates2; j ++)
	{
	  
	  modelState->setDiscreteState(col, j);

	  stateCollection->newModelState();
	  
	  double value = vFunction->getValue(stateCollection);

	  if (j < numDiscStates2-1)
	    fprintf(stream, "%f, ", value);
	  else
	    fprintf(stream, "%f\n", value);
	}
    }
}

void CVFunctionAnalyzer::saveDiscrete2DMatlab(FILE *stream, CState *initState, int row, int col)
{
  int numDiscStates1 = modelStateProperties->getDiscreteStateSize(row);
  int numDiscStates2 = modelStateProperties->getDiscreteStateSize(col);

  CState *modelState = stateCollection->getState(modelStateProperties);

  modelState->setState(initState);

  for (int i = 0; i < numDiscStates1; i ++)
  {

	modelState->setDiscreteState(row, i);

      	for (int j = 0; j < numDiscStates2; j ++)
	{
	  
	  	modelState->setDiscreteState(col, j);

	  	stateCollection->newModelState();
	  
	  	double value = vFunction->getValue(stateCollection);

	  	fprintf(stream, "%d %d %f\n", i, j, value);
	}
  }
}

void CVFunctionAnalyzer::saveStateValues(FILE *stream, CStateList *states)
{
	CState *modelState = stateCollection->getState(modelStateProperties);

	fprintf(stream, "VFunction Analyzer State Value-Table\n");

	for (unsigned int i = 0; i < states->getNumStates(); i ++)
	{
		states->getState(i, modelState);
		stateCollection->newModelState();

		double value = vFunction->getValue(stateCollection);
		fprintf(stream, "(");
		modelState->saveASCII(stream);
		fprintf(stream, ", %f)\n", value);
	}
}

void CVFunctionAnalyzer::save1DValues(char *filename, CState *initstate, int dim1, int part1)
{
	FILE *stream = fopen(filename, "w");
	
	if (stream == NULL)
	{
		fprintf(stderr, "VFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save1DValues(stream, initstate, dim1, part1);

	fclose(stream);
}
	
void CVFunctionAnalyzer::saveDiscrete1DValues(char *filename, CState *initstate, int dim1)
{
	FILE *stream = fopen(filename, "w");
	
	if (stream == NULL)
	{
		fprintf(stderr, "VFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete1DValues(stream, initstate, dim1);

	fclose(stream);
}
	

void CVFunctionAnalyzer::save2DValues(char *filename, CState *initstate, int dim1, int part1, int dim2, int part2)
{
	FILE *stream = fopen(filename, "w");
	
	if (stream == NULL)
	{
		fprintf(stderr, "VFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save2DValues(stream, initstate, dim1, part1, dim2, part2);

	fclose(stream);
}


void CVFunctionAnalyzer::saveDiscrete2DValues(char *filename, CState *initstate, int row, int col)
{
	FILE *stream = fopen(filename, "w");
	
	if (stream == NULL)
	{
		fprintf(stderr, "VFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete2DValues(stream, initstate, row, col);

	fclose(stream);
}

void CVFunctionAnalyzer::save1DMatlab(char *filename, CState *initstate, int dim1, int part1)
{
	FILE *stream = fopen(filename, "w");
	
	if (stream == NULL)
	{
		fprintf(stderr, "VFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save1DMatlab(stream, initstate, dim1, part1);

	fclose(stream);
}
	
void CVFunctionAnalyzer::saveDiscrete1DMatlab(char *filename, CState *initstate, int dim1)
{
	FILE *stream = fopen(filename, "w");
	
	if (stream == NULL)
	{
		fprintf(stderr, "VFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete1DMatlab(stream, initstate, dim1);

	fclose(stream);
}
	

void CVFunctionAnalyzer::save2DMatlab(char *filename, CState *initstate, int dim1, int part1, int dim2, int part2)
{
	FILE *stream = fopen(filename, "w");
	
	if (stream == NULL)
	{
		fprintf(stderr, "VFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save2DMatlab(stream, initstate, dim1, part1, dim2, part2);

	fclose(stream);
}


void CVFunctionAnalyzer::saveDiscrete2DMatlab(char *filename, CState *initstate, int row, int col)
{
	FILE *stream = fopen(filename, "w");
	
	if (stream == NULL)
	{
		fprintf(stderr, "VFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete2DMatlab(stream, initstate, row, col);

	fclose(stream);
}


void CVFunctionAnalyzer::saveStateValues(char *filename, CStateList *states)
{
	FILE *stream = fopen(filename, "w");
	
	if (stream == NULL)
	{
		fprintf(stderr, "VFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveStateValues(stream,states);

	fclose(stream);
}
	
void CVFunctionAnalyzer::setVFunction(CAbstractVFunction *l_vFunction)
{
	this->vFunction = l_vFunction;
}


CQFunctionAnalyzer::CQFunctionAnalyzer(CAbstractQFunction *qFunction, CStateProperties *modelState, std::list<CStateModifier *> *modifiers)
{
	this->qFunction = qFunction;
	this->modelStateProperties = modelState;

	stateCollection = new CStateCollectionImpl(modelState, modifiers);
}

CQFunctionAnalyzer::~CQFunctionAnalyzer()
{
	delete stateCollection;
}

void CQFunctionAnalyzer::setQFunction(CAbstractQFunction *l_qFunction)
{
	this->qFunction = l_qFunction;
}

void CQFunctionAnalyzer::save1DValues(FILE *stream, CActionSet *actions, CState *initState, int dim1, int part1)
{
	double width = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);
	fprintf(stream, "QFunction Analyzer 1D-Table for Dimension %d\n", dim1);
	fprintf(stream, "InitialState: ");
	initState->saveASCII(stream);
	fprintf(stream, "\n");

	fprintf(stream, "Actions: %d\n", actions->size());

	int index = 0;
	CActionSet::iterator it = actions->begin();
	
	for (; it != actions->end(); it ++)
    	{
      		fprintf(stream, "Action %d\n", index++);

      		for (int i = 0; i <= part1; i ++)
		{
			double x = modelStateProperties->getMinValue(dim1) + i * width / part1;
			modelState->setContinuousState(dim1, x);
			stateCollection->newModelState();
			fprintf(stream, "%f ", x);
			
			double value = qFunction->getValue(stateCollection, *it);
			fprintf(stream, "%f ", value);
			
			fprintf(stream, "\n");
		}
      		fprintf(stream, "\n\n");
    	}
	
}


void CQFunctionAnalyzer::save1DValues(char *filename, CActionSet *actions, CState *initState, int dim1, int part1) {
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save1DValues(file, actions, initState, dim1, part1);
	fclose(file);
}

void CQFunctionAnalyzer::save1DMaxAction(FILE *stream, CActionSet *actions, CState *initState, int dim1, int part1, char *actionSymbols)
{
	double width = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);
	fprintf(stream, "QFunction Analyzer 1D-Table for Dimension %d\n", dim1);
	fprintf(stream, "InitialState: ");
	initState->saveASCII(stream);
	fprintf(stream, "\n");

	fprintf(stream, "Actions: %d\n", actions->size());

	int index = 0;

	CActionSet availableActions;
	
	for (int i = 0; i <= part1; i ++)
	{
		double x = modelStateProperties->getMinValue(dim1) + i * width / part1;
		modelState->setContinuousState(dim1, x);
		stateCollection->newModelState();
		fprintf(stream, "%f ", x);
			
		availableActions.clear();
		actions->getAvailableActions(&availableActions, stateCollection);
		CAction *action = qFunction->getMax(stateCollection, &availableActions);

		if (actionSymbols == NULL)
		{
			fprintf(stream, "%d", actions->getIndex(action));
		}
		else
		{
			fprintf(stream, "%c", actionSymbols[actions->getIndex(action)]);
		}
			
		fprintf(stream, "\n");

    	}
	
}


void CQFunctionAnalyzer::save1DMaxAction(char *filename, CActionSet *actions, CState *initState, int dim1, int part1,  char *actionSymbols) {
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save1DMaxAction(file, actions, initState, dim1, part1, actionSymbols);
	fclose(file);
}


void CQFunctionAnalyzer::save1DMatlab(FILE *stream, CActionSet *actions, CState *initState, int dim1, int part1)
{
	double width = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);
	
	for (int i = 0; i <= part1; i ++)
	{
		double x = modelStateProperties->getMinValue(dim1) + i * width / part1;
		modelState->setContinuousState(dim1, x);
		stateCollection->newModelState();
		fprintf(stream, "%f ", x);
		CActionSet::iterator it = actions->begin();

		for (; it != actions->end(); it ++)
		{
			double value = qFunction->getValue(stateCollection, *it);
			fprintf(stream, "%f ", value);
		}
		fprintf(stream, "\n");
	}
}


void CQFunctionAnalyzer::save1DMatlab(char *filename, CActionSet *actions, CState *initState, int dim1, int part1) {
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save1DMatlab(file, actions, initState, dim1, part1);
	fclose(file);
}


void CQFunctionAnalyzer::saveDiscrete1DValues(FILE *stream, CActionSet *actions, CState *initState, int dim1)
{
  int numDiscStates = modelStateProperties->getDiscreteStateSize(dim1);

  CState *modelState = stateCollection->getState(modelStateProperties);

  modelState->setState(initState);

  fprintf(stream, "QFunction Analyzer 1D-Table for Dimension %d\n", dim1);
  fprintf(stream, "InitialState: ");
  initState->saveASCII(stream);
  fprintf(stream, "\n");
  
  fprintf(stream, "Actions: %d\n", actions->size());

  int index = 0;
  CActionSet::iterator it = actions->begin();
	
  for (; it != actions->end(); it ++)
  {
	
	fprintf(stream, "Action %d\n", index ++);

  	for (int i = 0; i < numDiscStates; i ++)
    	{
      		modelState->setDiscreteState(dim1, i);
      		stateCollection->newModelState();
      		
      		double value = qFunction->getValue(stateCollection, *it);
	  	fprintf(stream, "%d %f ",i, value);
	}
      	fprintf(stream, "\n");
   }
}


void CQFunctionAnalyzer::saveDiscrete1DValues(char *filename, CActionSet *actions, CState *initState, int dim1) {
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete1DValues(file, actions, initState, dim1);
	fclose(file);
}

void CQFunctionAnalyzer::saveDiscrete1DMaxAction(FILE *stream, CActionSet *actions, CState *initState, int dim1, char *actionSymbols)
{
  int numDiscStates = modelStateProperties->getDiscreteStateSize(dim1);

  CState *modelState = stateCollection->getState(modelStateProperties);

  modelState->setState(initState);

  fprintf(stream, "QFunction Analyzer 1D-Table for Dimension %d\n", dim1);
  fprintf(stream, "InitialState: ");
  initState->saveASCII(stream);
  fprintf(stream, "\n");
  
  fprintf(stream, "Actions: %d\n", actions->size());

  int index = 0;
  CActionSet availableActions;
	
   for (int i = 0; i < numDiscStates; i ++)
   {
   	modelState->setDiscreteState(dim1, i);
   	stateCollection->newModelState();
      		
   	availableActions.clear();
	actions->getAvailableActions(&availableActions, stateCollection);
	CAction *action = qFunction->getMax(stateCollection, &availableActions);

	if (actionSymbols == NULL)
	{
		fprintf(stream, "%d", actions->getIndex(action));
	}
	else
	{
		fprintf(stream, "%c", actionSymbols[actions->getIndex(action)]);
	}
    }
    fprintf(stream, "\n");
}


void CQFunctionAnalyzer::saveDiscrete1DMaxAction(char *filename, CActionSet *actions, CState *initState, int dim1, char *actionSymbols) {
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete1DMaxAction(file, actions, initState, dim1, actionSymbols);
	fclose(file);
}

void CQFunctionAnalyzer::saveDiscrete1DMatlab(FILE *stream, CActionSet *actions, CState *initState, int dim1)
{
  int numDiscStates = modelStateProperties->getDiscreteStateSize(dim1);

  CState *modelState = stateCollection->getState(modelStateProperties);

  modelState->setState(initState);

  int index = 0;
  for (int i = 0; i < numDiscStates; i ++)
  {
  	modelState->setDiscreteState(dim1, i);
  	stateCollection->newModelState();
      	
	fprintf(stream, "%d ", i);
	CActionSet::iterator it = actions->begin();
	
	for (; it != actions->end(); it ++)
  	{
  		double value = qFunction->getValue(stateCollection, *it);
	  	fprintf(stream,  "%f ",value);
	}
	fprintf(stream, "\n");
  }
      	
   
}


void CQFunctionAnalyzer::saveDiscrete1DMatlab(char *filename, CActionSet *actions, CState *initState, int dim1) {
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete1DMatlab(file, actions, initState, dim1);
	fclose(file);
}




void CQFunctionAnalyzer::save2DValues(char *filename, CActionSet *actions, CState *initstate, int dim1, int part1, int dim2, int part2)
{
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save2DValues(file, actions, initstate, dim1, part1, dim2, part2);
	fclose(file);
}

void CQFunctionAnalyzer::save2DValues(FILE *stream, CActionSet *actions, CState *initState, int dim1, int part1, int dim2, int part2)
{
	double width1 = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	double width2 = modelStateProperties->getMaxValue(dim2) - modelStateProperties->getMinValue(dim2);

	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);

	fprintf(stream, "QFunction Analyzer 2D-Table for Dimensions %d and %d\n", dim1, dim2);
	fprintf(stream, "InitialState: ");
	initState->saveASCII(stream);
	fprintf(stream, "\n");

	fprintf(stream, "Actions: %d\n", actions->size());

	CActionSet::iterator it = actions->begin();

	for (int index = 0; it != actions->end(); it ++, index++ )
	{
		fprintf(stream, "Action %d\n", index);
		for (int i = 0; i <= part1; i ++)
		{
			double x = modelStateProperties->getMinValue(dim1) + i * width1 / part1;
			modelState->setContinuousState(dim1, x);

			for (int j = 0; j <= part2; j ++)
			{
				double y = modelStateProperties->getMinValue(dim2) + j * width2 / part2;
				modelState->setContinuousState(dim2, y);
				stateCollection->newModelState();
							
				double value = qFunction->getValue(stateCollection, *it);
				fprintf(stream, "%f ", value);
			}

			fprintf(stream, "\n");
		}
	}
}

void CQFunctionAnalyzer::save2DMaxAction(char *filename, CActionSet *actions, CState *initstate, int dim1, int part1, int dim2, int part2, char *actionSymbols)
{
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save2DMaxAction(file, actions, initstate, dim1, part1, dim2, part2, actionSymbols);
	
	fclose(file);
}

void CQFunctionAnalyzer::save2DMaxAction(FILE *stream, CActionSet *actions, CState *initState, int dim1, int part1, int dim2, int part2, char *actionSymbols)
{
	double width1 = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	double width2 = modelStateProperties->getMaxValue(dim2) - modelStateProperties->getMinValue(dim2);

	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);

	fprintf(stream, "QFunction Analyzer 2D-Table for Dimensions %d and %d\n", dim1, dim2);
	fprintf(stream, "InitialState: ");
	initState->saveASCII(stream);
	fprintf(stream, "\n");

	fprintf(stream, "Actions: %d\n", actions->size());

	CActionSet availableActions;

	for (int i = 0; i <= part1; i ++)
	{
		double x = modelStateProperties->getMinValue(dim1) + i * width1 / part1;
		modelState->setContinuousState(dim1, x);

		for (int j = 0; j <= part2; j ++)
		{
			double y = modelStateProperties->getMinValue(dim2) + j * width2 / part2;
			modelState->setContinuousState(dim2, y);
			stateCollection->newModelState();
							
		   	availableActions.clear();
			actions->getAvailableActions(&availableActions, stateCollection);
			CAction *action = qFunction->getMax(stateCollection, &availableActions);

			if (actionSymbols == NULL)
			{
				fprintf(stream, "%d", actions->getIndex(action));
			}
			else
			{
				fprintf(stream, "%c", actionSymbols[actions->getIndex(action)]);
			}
		}

		fprintf(stream, "\n");
	}
}


void CQFunctionAnalyzer::save2DMatlab(char *filename, CActionSet *actions, CState *initstate, int dim1, int part1, int dim2, int part2)
{
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	save2DMatlab(file, actions, initstate, dim1, part1, dim2, part2);
	fclose(file);
}

void CQFunctionAnalyzer::save2DMatlab(FILE *stream, CActionSet *actions, CState *initState, int dim1, int part1, int dim2, int part2)
{
	double width1 = modelStateProperties->getMaxValue(dim1) - modelStateProperties->getMinValue(dim1);
	double width2 = modelStateProperties->getMaxValue(dim2) - modelStateProperties->getMinValue(dim2);

	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);

	for (int i = 0; i <= part1; i ++)
	{
		double x = modelStateProperties->getMinValue(dim1) + i * width1 / part1;
		modelState->setContinuousState(dim1, x);

		for (int j = 0; j <= part2; j ++)
		{
			double y = modelStateProperties->getMinValue(dim2) + j * width2 / part2;
			modelState->setContinuousState(dim2, y);
			stateCollection->newModelState();
			fprintf(stream, "%f %f ", x, y);

			CActionSet::iterator it = actions->begin();

			for (; it != actions->end(); it ++)
			{
				double value = qFunction->getValue(stateCollection, *it);
				fprintf(stream, "%f ", value);
			}

			fprintf(stream, "\n");
		}
	}
}


void CQFunctionAnalyzer::saveDiscrete2DValues(char *filename, CActionSet *actions, CState *initstate, int row_dim, int col_dim)
{
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete2DValues(file, actions, initstate, row_dim, col_dim);
	fclose(file);
}

void CQFunctionAnalyzer::saveDiscrete2DValues(FILE *stream, CActionSet *actions, CState *initState, int row_dim, int col_dim)
{

  int numDiscStates1 = modelStateProperties->getDiscreteStateSize(row_dim);
  int numDiscStates2 = modelStateProperties->getDiscreteStateSize(col_dim);

  CState *modelState = stateCollection->getState(modelStateProperties);

  modelState->setState(initState);

  fprintf(stream, "QFunction Analyzer 2D-Table for Dimensions [%d, %d]\n", row_dim, col_dim);
  fprintf(stream, "InitialState: ");
  initState->saveASCII(stream);
  fprintf(stream, "\n");
  
  fprintf(stream, "Actions: %d\n\n\n", actions->size());

  CActionSet::iterator it = actions->begin();
  
  int index = 0;

  for (; it != actions->end(); it ++)
    {
      fprintf(stream, "Action %d\n", index++);

      for (int i = 0; i < numDiscStates1; i ++)
	{
	  
	  modelState->setDiscreteState(row_dim, i);
	  
	  for (int j = 0; j < numDiscStates2; j ++)
	    {
	      
	      modelState->setDiscreteState(col_dim, j);
	      
	      stateCollection->newModelState();
	      
	      double value = qFunction->getValue(stateCollection, *it);
	      fprintf(stream, "%f ", value);
	      
	    }
	  fprintf(stream, "\n");

	}
      fprintf(stream, "\n\n");
    }
}

void CQFunctionAnalyzer::saveDiscrete2DMaxAction(char *filename, CActionSet *actions, CState *initstate, int row_dim, int col_dim, char *actionSymbols)
{
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete2DMaxAction(file, actions, initstate, row_dim, col_dim, actionSymbols);
	fclose(file);
}

void CQFunctionAnalyzer::saveDiscrete2DMaxAction(FILE *stream, CActionSet *actions, CState *initState, int row_dim, int col_dim, char *actionSymbols)
{

  int numDiscStates1 = modelStateProperties->getDiscreteStateSize(row_dim);
  int numDiscStates2 = modelStateProperties->getDiscreteStateSize(col_dim);

  CState *modelState = stateCollection->getState(modelStateProperties);

  modelState->setState(initState);

  fprintf(stream, "QFunction Analyzer 2D-Table for Dimensions [%d, %d]\n", row_dim, col_dim);
  fprintf(stream, "InitialState: ");
  initState->saveASCII(stream);
  fprintf(stream, "\n");
  
  fprintf(stream, "Actions: %d\n\n\n", actions->size());

  CActionSet availableActions;
  
  int index = 0;

  fprintf(stream, "Action %d\n", index++);

  for (int i = 0; i < numDiscStates1; i ++)
  {
	  
	  modelState->setDiscreteState(row_dim, i);
	  
	  for (int j = 0; j < numDiscStates2; j ++)
	    {
	      
	      modelState->setDiscreteState(col_dim, j);
	      
	      stateCollection->newModelState();
	      
	   	availableActions.clear();
		actions->getAvailableActions(&availableActions, stateCollection);
		CAction *action = qFunction->getMax(stateCollection, &availableActions);

		if (actionSymbols == NULL)
		{
			fprintf(stream, "%d", actions->getIndex(action));
		}
		else
		{
			fprintf(stream, "%c", actionSymbols[actions->getIndex(action)]);
		}
	      
	    }
	  fprintf(stream, "\n");

   }
 
}

void CQFunctionAnalyzer::saveDiscrete2DMatlab(char *filename, CActionSet *actions, CState *initstate, int dim1, int dim2)
{
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveDiscrete2DMatlab(file, actions, initstate, dim1, dim2);
	fclose(file);
}

void CQFunctionAnalyzer::saveDiscrete2DMatlab(FILE *stream, CActionSet *actions, CState *initState, int dim1, int dim2)
{
	CState *modelState = stateCollection->getState(modelStateProperties);

	modelState->setState(initState);

	for (int i = 0; i < modelStateProperties->getDiscreteStateSize(dim1); i ++)
	{
		
		modelState->setDiscreteState(dim1, i);

		for (int j = 0; j < modelStateProperties->getDiscreteStateSize(dim2); j ++)
		{
			modelState->setDiscreteState(dim2,j);
			stateCollection->newModelState();
			fprintf(stream, "%d %d ", i, j);

			CActionSet::iterator it = actions->begin();

			for (; it != actions->end(); it ++)
			{
				double value = qFunction->getValue(stateCollection, *it);
				fprintf(stream, "%f ", value);
			}

			fprintf(stream, "\n");
		}
	}
}


void CQFunctionAnalyzer::saveStateValues(FILE *stream, CActionSet *actions, CStateList *states)
{
	CState *modelState = stateCollection->getState(modelStateProperties);

	fprintf(stream, "QFunction Analyzer State Value-Table\n");

	for (unsigned int i = 0; i < states->getNumStates(); i ++)
	{
		states->getState(i, modelState);
		stateCollection->newModelState();

		fprintf(stream, "(");
		modelState->saveASCII(stream);
		fprintf(stream, ": ");

		CActionSet::iterator it = actions->begin();

		for (; it != actions->end(); it ++)
		{
			double value = qFunction->getValue(stateCollection, *it);
			fprintf(stream, "%f ", value);
		}
		fprintf(stream, ")\n");
	}
}

void CQFunctionAnalyzer::saveStateValues(char *filename, CActionSet *actions, CStateList *states) {
	FILE *file = fopen(filename, "w");
	
	if (file == NULL)
	{
		fprintf(stderr, "QFunctionAnalyzer: File not found %s!!\n", filename);
		return;
	}
	
	saveStateValues(file, actions, states);
	fclose(file);
}



CFunctionComperator::CFunctionComperator(CStateProperties *modelState, std::list<CStateModifier *> *modifiers)
{
	this->modelStateProperties = modelState;
	stateCollection = new CStateCollectionImpl(modelStateProperties, modifiers);
}

CFunctionComperator::~CFunctionComperator()
{
	delete stateCollection;
}

double CFunctionComperator::getDifference(CStateCollection *state, int errorFunction)
{
	double value1 = getValue(1, state);
	double value2 = getValue(2, state);

	double error = 0;

	switch (errorFunction)
	{
	case ANALYZER_MSE:
		{
			error = pow(value1 - value2, 2);
			break;
		}
	case ANALYZER_MAE:
		{
			error = fabs((double)(value1 - value2));
		}
	}
	return error;
}

void CFunctionComperator::getRandomState(CState *state)
{
	for (unsigned int i = 0; i < state->getNumContinuousStates(); i ++)
	{
		double width = state->getStateProperties()->getMaxValue(i) - state->getStateProperties()->getMinValue(i);
		double randVal = ((double) rand() / (double) RAND_MAX) * width + state->getStateProperties()->getMaxValue(i);

		state->setContinuousState(i, randVal);
	}
}

double CFunctionComperator::compareFunctionsRandom(int nSamples, int errorFunction )
{
	CState *modelState = stateCollection->getState(modelStateProperties);
	double error = 0.0;
	for (int i = 0; i < nSamples; i++)
	{
		getRandomState(modelState);
		stateCollection->newModelState();

		switch(errorFunction)
		{
		case ANALYZER_MSE:
		case ANALYZER_MAE:
			{
				double value = getDifference(stateCollection, errorFunction);
				error += value;
				break;
			}
		case ANALYZER_MAXERROR:
			{
				double value = getDifference(stateCollection, ANALYZER_MAE);
				if (value > error)
				{
					error = value;
				}
				break;
			}

		}
	}
	if (errorFunction == ANALYZER_MSE || errorFunction == ANALYZER_MAE)
	{
		error /= nSamples;
	}
	return error;
}

double CFunctionComperator::compareFunctionsStates(CStateList *states, int errorFunction)
{
	CState *modelState = stateCollection->getState(modelStateProperties);
	double error = 0.0;
	for (unsigned int i = 0; i < states->getNumStates(); i++)
	{
		states->getState(i, modelState);
		stateCollection->newModelState();
		switch(errorFunction)
		{
		case ANALYZER_MSE:
		case ANALYZER_MAE:
			{
				double value = getDifference(stateCollection, errorFunction);
				error += value;
				break;
			}
		case ANALYZER_MAXERROR:
			{
				double value = getDifference(stateCollection, ANALYZER_MAE);
				if (value > error)
				{
					error = value;
				}
				break;
			}

		}
	}
	if (errorFunction == ANALYZER_MSE || errorFunction == ANALYZER_MAE)
	{
		error /= states->getNumStates();
	}
	return error;
}

double CFunctionComperator::getReward(CStateCollection *oldState, CAction *, CStateCollection *)
{
	return getDifference(oldState, ANALYZER_MAE);
}

double CVFunctionComperator::getValue(int numFunc, CStateCollection *state)
{
	if (numFunc == 1)
	{
		return vFunction1->getValue(state);
	}
	else
	{
		return vFunction2->getValue(state);
	}
}

CVFunctionComperator::CVFunctionComperator(CStateProperties *modelState, std::list<CStateModifier *> *modifiers, CAbstractVFunction *vFunction1, CAbstractVFunction *vFunction2) : CFunctionComperator(modelState, modifiers)
{
	this->vFunction1 = vFunction1;
	this->vFunction2 = vFunction2;
}

double CQFunctionComperator::getValue(int numFunc, CStateCollection *state)
{
	if (numFunc == 1)
	{
		return qFunction1->getValue(state, action);
	}
	else
	{
		return qFunction2->getValue(state, action);
	}
}

CQFunctionComperator::CQFunctionComperator(CStateProperties *modelState, std::list<CStateModifier *> *modifiers, CAbstractQFunction *qFunction1, CAbstractQFunction *qFunction2, CAction *action) : CFunctionComperator(modelState, modifiers)
{
	this->qFunction1 = qFunction1;
	this->qFunction2 = qFunction2;

	this->action = action;
}


CControllerAnalyzer::CControllerAnalyzer(CStateList *states, CAgentController *controller, CActionSet *actions) : CActionObject(actions)
{
	this->states = states;
	this->controller = controller;
}

CControllerAnalyzer::~CControllerAnalyzer()
{
}

CStateList *CControllerAnalyzer::getStateList()
{
	return this->states;
}

void CControllerAnalyzer::setStateList(CStateList *states)
{
	this->states = states;
}

CAgentController *CControllerAnalyzer::getController()
{
	return this->controller;
}

void CControllerAnalyzer::setController(CAgentController *controller)
{
	this->controller = controller;
}

void CControllerAnalyzer::saveActions(FILE *stream, std::list<CStateModifier *> *modifiers)
{
	fprintf(stream, "State - selected Action Table\n");
	CState *modelState = new CState(states->getStateProperties());
	CAction *action;
    unsigned int i = 1;
	CStateCollectionImpl *stateCollection = new CStateCollectionImpl(states->getStateProperties(), modifiers);

	for (unsigned int stateNum = 0; stateNum < states->getNumStates(); stateNum ++)
	{
		states->getState(stateNum, modelState);
		stateCollection->setState(modelState);
		stateCollection->calculateModifiedStates();

		action = controller->getNextAction(stateCollection);
		assert(action != NULL);
		fprintf(stream, "State# %d: Action %d\n", i++, actions->getIndex(action));
	}
	delete modelState;
	delete stateCollection;
}
/*

CActionStatisticAnalyzer::CActionStatisticAnalyzer(CAgentStatisticController *master) : CAgentStatisticController(NULL)
{
	this->master = master;
	controllers = new std::map<void *, std::pair<char *, long>*>();
}

CActionStatisticAnalyzer::~CActionStatisticAnalyzer()
{
	if (controllers->size() > 0)
	{
		std::map<void *, std::pair<char *, long>*>::iterator it = controllers->begin();
        for(;it != controllers->end(); it++)
		{
			delete (*it).second;
		}
		controllers->clear();
	}
    delete controllers;
}

void CActionStatisticAnalyzer::addController(CAgentStatisticController *newcontroller, char *name)
{
	std::pair<char *, long>* temp = new std::pair<char *, long>(name, 0);
	(*controllers)[(void *)newcontroller] = temp;
}

void CActionStatisticAnalyzer::printStatistics()
{
	if (controllers->size() > 0)
	{
		printf("ActionOwners: ");
		std::map<void *, std::pair<char *, long>*>::iterator it = controllers->begin();
        for(;it != controllers->end(); it++)
		{
			printf("%s: %ld; ", (*it).second->first, (*it).second->second);
		}
		printf("\n");
	}
}

void CActionStatisticAnalyzer::init()
{
	if (controllers->size() > 0)
	{
		std::map<void *, std::pair<char *, long>*>::iterator it = controllers->begin();
        for(;it != controllers->end(); it++)
		{
			(*it).second->second = 0;
		}
	}
}

CAction* CActionStatisticAnalyzer::getNextAction(CStateCollection *state, CActionStatistics *stat)
{
	CAction *action = master->getNextAction(state, NULL, stat);
	assert (stat != NULL);
    if (controllers->find(stat->owner) != controllers->end())
	{
		(*controllers)[(void *)stat->owner]->second ++;
	}
	return action;
}
*/

CFittedQIterationAnalyzer::CFittedQIterationAnalyzer(CQFunction *qFunction, CAgentController *estimationPolicy, CEpisodeHistory *episodeHistory, CRewardHistory *rewardLogger, CSupervisedQFunctionLearner *learner, CStateProperties *residualProperties, CPolicySameStateEvaluator *l_evaluator) : CFittedQIteration(qFunction, estimationPolicy, episodeHistory, rewardLogger, learner, residualProperties), CTestSuiteEvaluatorLogger("Analyzer")
{
	evaluator = l_evaluator;
	analyzerFile = NULL;

	kNN = 5;

	buffState2 = new CState(residualProperties);

	addParameter("UseRealQValues", 0.0);

	useQValues = false;

	lastQValue = 0;
	lastEstimatedQValue = 0;
}

CFittedQIterationAnalyzer::~CFittedQIterationAnalyzer()
{
	delete buffState2;
}

void CFittedQIterationAnalyzer::addResidualInput(CStep *step, CAction *nextAction, double V, double newV, double nearestNeighborDistance, CAction *, double )
{
	if (!useQValues)
	{
		CFittedQIteration::addResidualInput(step, nextAction, V,  newV, nearestNeighborDistance);
		
	}

	
	/*if (nextAction == NULL)
	{
		printf("%d reward: %f\n", episodeHistory->getActions()->getIndex(step->action), nextReward);
	
		printf("Real States:\n");
		step->oldState->getState()->saveASCII( stdout);
		printf("\n");

		step->newState->getState()->saveASCII( stdout);
		printf("\n");

		double estR = evaluator->getActionValueForState(step->oldState->getState(), step->action, numEvaluations);

		printf("Estimated Reward: %f\n", estR);	
	}*/


	
	
	
	

	if (nextAction != NULL && analyzerFile && (*nearestNeighbors)[nextAction] != NULL)
	{
		CStateProperties *l_prop = residualProperties; 

		if (l_prop == NULL)
		{
			CBatchQDataGenerator *qGenerator = dynamic_cast<CBatchQDataGenerator *>(dataGenerator);
				
			l_prop = qGenerator->getStateProperties(nextAction);
		}
		
		
		/*CState *state1 = step->oldState->getState(l_prop);

		CState newState(state1);

		(*dataPreProc)[step->action]->preprocessInput(state1, &newState);

		neighborsList->clear();
		(*nearestNeighbors)[step->action]->getNearestNeighbors(state1, neighborsList);
			
		std::list<int>::iterator it1 = neighborsList->begin();
		for (; it1 != neighborsList->end(); it1 ++)
		{
			ColumnVector *nearestNeighbor = (*(*inputDatas)[step->action])[*(it1)];
			
			double distance = newState.getDistance(nearestNeighbor);	
	
			printf("%f ", distance);
		}
		printf("\n");*/
		
		/*double realV = 

		if (useQValues)
		{
			dataGenerator->addInput(step->newState, nextAction, realV);
		}*/

		int actionIndex1 = episodeHistory->getActions()->getIndex(step->action);
		int actionIndex2 = episodeHistory->getActions()->getIndex(nextAction);
		fprintf(analyzerFile, "%d %d %f %f ", actionIndex1, actionIndex2, lastEstimatedQValue, lastQValue);

/*
		if (nextHistoryAction != NULL)
		{
			fprintf(analyzerFile, "%d %f ", episodeHistory->getActions()->getIndex(nextHistoryAction), nextReward);
		}
		else
		{
			fprintf(analyzerFile, "-1 0.0 ");
		}*/

		CState *state = step->newState->getState(l_prop);
		(*dataPreProc)[nextAction]->preprocessInput(state, buffState);
	
		
		neighborsList->clear();
		(*nearestNeighbors)[nextAction]->getNearestNeighbors(state, neighborsList);
		
		std::list<int>::iterator it = neighborsList->begin();
		for (; it != neighborsList->end(); it ++)
		{
			ColumnVector *nearestNeighbor = (*(*inputDatas)[nextAction])[*(it)];
		
			double distance = buffState->getDistance(nearestNeighbor);	

			fprintf(analyzerFile, "%f %f ", distance, (*(*outputDatas)[nextAction])[*it]);
		}

		CState *oldState = step->oldState->getState(l_prop);
		(*dataPreProc)[nextAction]->preprocessInput(oldState, buffState2);	

		it = neighborsList->begin();
		for (; it != neighborsList->end(); it ++)
		{
			ColumnVector *nearestNeighbor = (*(*inputDatas)[nextAction])[*(it)];
		
			double distance = buffState2->getDistance(nearestNeighbor);	

			fprintf(analyzerFile, "%f %f ", distance, (*(*outputDatas)[nextAction])[*it]);
		}

		double distance = buffState2->getDistance(buffState);
		fprintf(analyzerFile, "%f %f", distance, V);
		
		fprintf(analyzerFile, "\n");
	}
}


void CFittedQIterationAnalyzer::evaluate(string evaluationDirectory, int trial, int numEpisodes)
{
	stringstream filename;

	filename  << evaluationDirectory << outputDirectory << "/" << "Trial" << trial << "/FittedQAnalyzer_Eval" << numEpisodes << ".csv";

	if (analyzerFile)
	{
		fclose(analyzerFile);
	}

	analyzerFile = fopen(filename.str().c_str(), "w");

	numEvaluations = numEpisodes;

	int useRealQValues = (int) getParameter("UseRealQValues");

	useQValues = (useRealQValues > 0 && (numEvaluations % useRealQValues == 0));
}
	
void CFittedQIterationAnalyzer::startNewEvaluation(string evaluationDirectory, CParameters *, int trial)
{
	stringstream sstream;
	
	sstream << "mkdir " << evaluationDirectory << outputDirectory << "/" << "Trial" << trial;
	string scall = sstream.str();
	system(scall.c_str());

		
	if (analyzerFile)
	{
		fclose(analyzerFile);
	}
	analyzerFile = NULL;
	numEvaluations = 0;
}


double CFittedQIterationAnalyzer::getValue(CStateCollection *stateCollection, CAction *action)
{
	lastEstimatedQValue = CFittedQIteration::getValue(stateCollection, action);
	

	lastQValue = evaluator->getActionValueForState(stateCollection->getState(), action, numEvaluations);

	if (useQValues)
	{
		return lastQValue;
	}
	else
	{
		return lastEstimatedQValue;
	}
}
