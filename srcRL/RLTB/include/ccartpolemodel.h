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

#ifndef __CCARTPOLE_H
#define __CCARTPOLE_H

#include "cqtconfig.h"

#include "ctransitionfunction.h"
#include "crewardfunction.h"


#ifdef RL_TOOLBOX_USE_QT
#include "cqtmodelvisualizer.h"
#endif

class CCartPoleModel : public CLinearActionContinuousTimeTransitionFunction
{
protected:
	virtual void doSimulationStep(CState *state, double timestep, CAction *action, CActionData *data);


public:
	double uMax;
	double lengthTrack;
	double g;
	double massCart;
	double massPole;
	double lengthPole;
	double mu_c; // friction cart
	double mu_p; // friction pole

	bool endLeaveTrack;
	bool endOverRotate;

	double failedReward;

	CCartPoleModel(double dt, double uMax = 10, double lengthTrack = 4.8, double lengthPole = 0.5, double massCart = 1.0, double massPole = 0.5,  double mu_c = 1.0, double mu_p = 0.1, double g = 9.8, bool endLeaveTrack = true,bool endOverRotate = true);
	~CCartPoleModel();

	virtual Matrix *getB(CState *state);
	virtual ColumnVector *getA(CState *state);

	virtual bool isFailedState(CState *state);

	virtual void getResetState(CState *state);
};

class CCartPoleRewardFunction : public CStateReward
{
protected:
	CCartPoleModel *cartpoleModel;
public:
	bool useHeighPeak;
	bool punishOverRotate;
	CCartPoleRewardFunction(CCartPoleModel *model);
	~CCartPoleRewardFunction() {};

	virtual double getStateReward(CState *state);
	virtual void getInputDerivation(CState *modelState, ColumnVector *targetState);

};

class CCartPoleHeightRewardFunction : public CStateReward
{
protected:
	CCartPoleModel *cartpoleModel;
public:
	CCartPoleHeightRewardFunction(CCartPoleModel *model);

	virtual double getStateReward(CState *state);
	virtual void getInputDerivation(CState *modelState, ColumnVector *targetState);

};

#ifdef RL_TOOLBOX_USE_QT

class CQTCartPoleVisualizer : public CQTModelVisualizer
{
protected:
	double phi;
	double dphi;
	double x;
	double dx;

	CCartPoleModel *cartModel;

	virtual void doDrawState( QPainter *painter);

public:

	CQTCartPoleVisualizer( CCartPoleModel *model, QWidget *parent = NULL, const char *name = NULL);

	virtual void newDrawState(CStateCollection *state);
};

#endif

#endif

