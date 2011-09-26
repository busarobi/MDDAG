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

#ifndef C_LSTD_H
#define C_LSTD_H

#include "clearndataobject.h"
#include "cagentlistener.h"
#include "csupervisedlearner.h"

#include <stdlib.h>
#include <stdio.h>



class Matrix;
class ColumnVector;

class CFeatureVFunction;
class CFeatureQFunction;
class CStateProperties;
class CFeatureVETraces;
class CFeatureQETraces;
class CFeatureList;

class CGradientQETraces;
class CAgentController;
class CActionDataSet;


class CLSTDLambda : public CSemiMDPRewardListener, public CLearnDataObject, public CLeastSquaresLearner
{
protected:
	//CFeatureVFunction *vFunction;
	
	//CFeatureVETraces *vETraces;
	CFeatureList *oldStateGradient;
	CFeatureList *newStateGradient;
	
		
	int nEpisode;
	
	
	virtual void getOldGradient(CStateCollection *stateCol, CAction *action, CFeatureList *gradient) = 0;
	virtual void getNewGradient(CStateCollection *stateCol, CFeatureList *gradient) = 0;
	
	virtual void updateETraces(CStateCollection *stateCol, CAction *action) = 0;
	virtual CFeatureList *getGradientETraces() = 0;
	virtual void resetETraces() = 0;
	
public:
	int nUpdateEpisode;

	CLSTDLambda(CRewardFunction *rewardFunction, CGradientUpdateFunction *updateFunction, int nUpdatePerEpisode);
	virtual ~CLSTDLambda();
	
	virtual void nextStep(CStateCollection *oldState, CAction *action, double reward, CStateCollection *newState);
	virtual void newEpisode();
	
	
	virtual void resetData();
	virtual void loadData(FILE *stream);
	virtual void saveData(FILE *stream);
};

class CVLSTDLambda : public CLSTDLambda
{
	protected:
		CFeatureVFunction *vFunction;
		CFeatureVETraces *vETraces;
	
		virtual void getOldGradient(CStateCollection *stateCol, CAction *action, CFeatureList *gradient);
		virtual void getNewGradient(CStateCollection *stateCol, CFeatureList *gradient);
	
		virtual void updateETraces(CStateCollection *stateCol, CAction *action);
		virtual CFeatureList *getGradientETraces();
		virtual void resetETraces();
	public:
		CVLSTDLambda(CRewardFunction *rewardFunction, CFeatureVFunction *updateFunction, int nUpdatePerEpisode);
		virtual ~CVLSTDLambda();	
};

class CQLSTDLambda : public CLSTDLambda
{
	protected:
		CFeatureQFunction *qFunction;
		CGradientQETraces *qETraces;
		
		CAgentController *policy;
		CActionDataSet *actionDataSet;
	
		virtual void getOldGradient(CStateCollection *stateCol, CAction *action, CFeatureList *gradient);
		virtual void getNewGradient(CStateCollection *stateCol, CFeatureList *gradient);
	
		virtual void updateETraces(CStateCollection *stateCol, CAction *action);
		virtual CFeatureList * getGradientETraces();
		virtual void resetETraces();
	public:
		CQLSTDLambda(CRewardFunction *rewardFunction, CFeatureQFunction *updateFunction, CAgentController *policy,  int nUpdatePerEpisode);
		virtual ~CQLSTDLambda();	
};


#endif


