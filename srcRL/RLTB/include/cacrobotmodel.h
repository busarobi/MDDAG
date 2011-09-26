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

#ifndef __CACROBOT_H
#define __CACROBOT_H

#include "cqtconfig.h"

#include "ctransitionfunction.h"
#include "crewardfunction.h"
#include "ril_debug.h"

#ifdef RL_TOOLBOX_USE_QT
#include "cqtmodelvisualizer.h"
#endif

class CAcroBotModel : public CLinearActionContinuousTimeTransitionFunction
{
protected:
	virtual void doSimulationStep(CState *state, double timestep, CAction *action, CActionData *data);

public:
	double uMax;
	double g;
	double mass1;
	double mass2;
	double length1;
	double length2;
	double mu_1; 
	double mu_2; 

	CAcroBotModel(double dt, double uMax = 2, double length1 = 0.5, double length2 = 0.5, double mass1 = 1.0, double mass2 = 1.0, double mu_1 = 0.05, double mu_2 = 0.05, double g = 9.8);
	virtual ~CAcroBotModel();

	virtual Matrix *getB(CState *state);
	virtual ColumnVector *getA(CState *state);


	virtual bool isFailedState(CState *state);

	virtual void getResetState(CState *state);
};

/*
class CAcroBotModelSutton : public CAcroBotModel
{
protected:

public:

	double I1; 
	double I2; 

	CAcroBotModelSutton(double dt, double uMax = 2, double length1 = 1.0, double length2 = 1.0, double mass1 = 1.0, double mass2 = 1.0, double I1 = 1.0, double I2 = 1.0, double g = 9.8);
	virtual ~CAcroBotModelSutton();

	virtual Matrix *getB(CState *state);
	virtual ColumnVector *getA(CState *state);

};
*/

class CAcroBotRewardFunction : public CStateReward
{
protected:
	CAcroBotModel *model;
public:
	CAcroBotRewardFunction(CAcroBotModel *model, double segmentFactor = 0.5);
	virtual ~CAcroBotRewardFunction(){};
	
	double segmentFactor;
	bool useHeighPeak;

	double power;

	virtual double getStateReward(CState *state);
	virtual void getInputDerivation(CState *modelState, ColumnVector *targetState);

};

class CAcroBotHeightRewardFunction : public CStateReward
{
protected:
	CAcroBotModel *model;
	bool useHeighPeak;
public:
	CAcroBotHeightRewardFunction(CAcroBotModel *model);
	virtual ~CAcroBotHeightRewardFunction() {};

	virtual double getStateReward(CState *state);
	virtual void getInputDerivation(CState *modelState, ColumnVector *targetState);

};

class CAcroBotVelocityRewardFunction : public CStateReward
{
protected:
	CAcroBotModel *model;
public:
	bool invertVelocity;

	CAcroBotVelocityRewardFunction(CAcroBotModel *model);
	virtual ~CAcroBotVelocityRewardFunction(){};

	virtual double getStateReward(CState *state);
	virtual void getInputDerivation(CState *modelState, ColumnVector *targetState);
};

class CAcroBotExpRewardFunction : public CStateReward
{
protected:
	CAcroBotModel *model;
public:
	double expFactor;

	CAcroBotExpRewardFunction(CAcroBotModel *model, double expFactor = 10.0);
	virtual ~CAcroBotExpRewardFunction(){};

	virtual double getStateReward(CState *state);
	virtual void getInputDerivation(CState *modelState, ColumnVector *targetState);
};

#ifdef RL_TOOLBOX_USE_QT

class CQTAcroBotVisualizer : public CQTModelVisualizer
{
protected:
	double phi1;
	double dphi1;
	double phi2;
	double dphi2;

	CAcroBotModel *acroModel;

	virtual void doDrawState( QPainter *painter);

public:
	CQTAcroBotVisualizer( CAcroBotModel *acroModel, QWidget *parent=0, const char *name=0);
	virtual ~CQTAcroBotVisualizer() {};

	virtual void newDrawState(CStateCollection *state);
};

#endif

#endif


