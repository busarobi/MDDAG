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

#ifndef __CPENDULUMMODEL_H
#define __CPENDULUMMODEL_H

#include "cqtconfig.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ctransitionfunction.h"
#include "crewardfunction.h"
#include "cagentlistener.h"

#ifdef RL_TOOLBOX_USE_QT
#include "cqtmodelvisualizer.h"
#endif

class CPendulumModel : public CLinearActionContinuousTimeTransitionFunction
{
protected:
	virtual void doSimulationStep(CState *state, double timestep, CAction *action, CActionData *data);

public:
	double uMax;
	double dPhiMax;
	double g;
	double mass;
	double length;
	double mu; // friction

	CPendulumModel(double dt, double uMax = 5, double dPhiMax = 10, double length = 1, double mass = 1, double mu = 1.0, double g = 9.81);
	~CPendulumModel();

	virtual Matrix *getB(CState *state);
	virtual ColumnVector *getA(CState *state);

	virtual bool isFailedState(CState *state);



	virtual void getResetState(CState *resetState);

	virtual void setParameter(string paramName, double value);

};

class CPendulumRewardFunction : public CStateReward
{
public:
	double rewardFactor;
	CPendulumRewardFunction(CPendulumModel *model);

	virtual double getStateReward(CState *state);
	virtual void getInputDerivation(CState *modelState, ColumnVector *targetState);

};

class CPendulumUpTimeCalculator : public CSemiMDPListener
{
protected:
	double phi_up;
	double dt;
	int up_steps;
public:
	CPendulumUpTimeCalculator(double phi_up, double dt);

	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);
	virtual void newEpisode();

	double getUpTime();
	int getUpSteps();
};

/*
class CTestSuitePendulumUpTimeCalculatorEvaluator : public CTestSuiteEpisodesToLearnEvaluator
{
protected:
	int neededUpSteps;
	CPendulumUpTimeCalculator *upTimeCalc;

	virtual bool isEpisodeSuccessFull(FILE *stream);
public:
	CTestSuitePendulumUpTimeCalculatorEvaluator(CAgent *agent, int neededSuccEpisodes, int maxEpisodes, int stepsPerEpisode, int neededUpSteps,double phi_up);
	~CTestSuitePendulumUpTimeCalculatorEvaluator();
};*/

#ifdef RL_TOOLBOX_USE_QT

class CQTPendulumVisualizer : public CQTModelVisualizer
{
protected:
	double phi;
	double dphi;

	CPendulumModel *pendModel;

	virtual void doDrawState( QPainter *painter);

public:
	CQTPendulumVisualizer(CPendulumModel *pendModel, QWidget *parent=0, const char *name=0);

	virtual void newDrawState(CStateCollection *state);
};

#endif

#endif
