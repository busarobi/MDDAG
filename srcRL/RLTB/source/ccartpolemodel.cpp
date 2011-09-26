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

#include "ccartpolemodel.h"
#include "ril_debug.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "cstate.h"
#include "cstateproperties.h"
#include "caction.h"
#include "ccontinuousactions.h"

CCartPoleModel::CCartPoleModel(double dt, double uMax, double lengthTrack, double lengthPole, double massCart, double massPole, double mu_c, double mu_p, double g, bool endLeaveTrack, bool endOverRotate) : CLinearActionContinuousTimeTransitionFunction(new CStateProperties(5,0), new CContinuousAction(new CContinuousActionProperties(1)), dt)
{
	this->uMax = uMax;
	this->lengthTrack = lengthTrack;
	this->lengthPole = lengthPole;
	this->massCart = massCart;
	this->massPole = massPole;
	this->mu_c = mu_c;
	this->mu_p = mu_p;
	this->g = g;

	properties->setMaxValue(0, lengthTrack / 2 + 0.1);
	properties->setMinValue(0, -lengthTrack / 2 - 0.1);

	properties->setMaxValue(1, lengthTrack);
	properties->setMinValue(1, -lengthTrack);

	properties->setMaxValue(2, M_PI);
	properties->setMinValue(2, -M_PI);

	properties->setPeriodicity(2, true);

	properties->setMaxValue(3, 15);
	properties->setMinValue(3, -15);


	properties->setMaxValue(4, M_PI * 11);
	properties->setMinValue(4, -M_PI * 11);


	actionProp->setMaxActionValue(0,uMax);
	actionProp->setMinActionValue(0,-uMax);

	this->endLeaveTrack = endLeaveTrack;
	this->endOverRotate = endOverRotate;

	failedReward = -100.0;
}
	
CCartPoleModel::~CCartPoleModel()
{
	delete properties;
	delete actionProp;
	delete contAction;
}

Matrix *CCartPoleModel::getB(CState *state)
{
	double Phi, x, dx, dPhi;

	x = state->getContinuousState(0);
	dx = state->getContinuousState(1);
	Phi = state->getContinuousState(2);
	dPhi = state->getContinuousState(3);

	double denum = - 4 * lengthPole / 3 * (massCart + massPole) + lengthPole * massPole * pow(cos(Phi),2);
	double uFactordx = - 4 * lengthPole / denum / 3;
	double uFactordPhi =  - cos(Phi) / denum;


	*B = 0.0;
	B->element(1, 0) = uFactordx;
	B->element(3, 0) = uFactordPhi;


	return B;
}

ColumnVector *CCartPoleModel::getA(CState *state)
{
	double Phi, x, dx, ddx, dPhi, ddPhi;

	x = state->getContinuousState(0);
	dx = state->getContinuousState(1);
	Phi = state->getContinuousState(2);
	dPhi = state->getContinuousState(3);
	
	double sign_dx = 1.0;
	if (dx < 0)
	{
		sign_dx = - 1.0;
	}
	// Calculate Inverse Matrix

	double denum = - 4 * lengthPole / 3 * (massCart + massPole) + lengthPole * massPole * pow(cos(Phi),2);
	double b1 = g * sin(Phi) - mu_p * dPhi / (lengthPole * massPole);
	double b2 = lengthPole * massPole * dPhi * dPhi * sin(Phi) + mu_c * sign_dx;

	ddx = - lengthPole * massPole * cos(Phi) * b1 + 4 / 3 * lengthPole * b2;
	ddx = ddx / denum;

	ddPhi = (- massCart - massPole) * b1 + cos(Phi) * b2;
	ddPhi = ddPhi / denum;

	A->element(0) = dx;
	A->element(1) = ddx;
	A->element(2) = dPhi;
	A->element(3) = ddPhi;
	A->element(4) = dPhi;

	return A;
}

bool CCartPoleModel::isFailedState(CState *state)
{
	bool failed = (endLeaveTrack && fabs(state->getContinuousState(0)) > lengthTrack / 2);
	
	failed = failed | (endOverRotate && fabs(state->getContinuousState(4)) > 10 * M_PI);
	
	return  failed;
}

void CCartPoleModel::doSimulationStep(CState *state, double timestep, CAction *action, CActionData *data)
{
	getDerivationX(state, action, derivation, data);

	double ddx = derivation->element(1);
	double ddPhi = derivation->element(3);

	for (unsigned int i = 0; i < state->getNumContinuousStates(); i++)
	{
		state->setContinuousState(i, state->getContinuousState(i) + timestep * derivation->element(i));
	}

	state->setContinuousState(0, state->getContinuousState(0) + pow(timestep, 2) * ddx / 2);
	state->setContinuousState(2, state->getContinuousState(2) + pow(timestep, 2) * ddPhi / 2);
	state->setContinuousState(4, state->getContinuousState(4) + pow(timestep, 2) * ddPhi / 2);

	if (!endLeaveTrack && fabs(state->getContinuousState(0)) >= lengthTrack / 2 )
	{
		state->setContinuousState(1, 0.0);
	}
}

void CCartPoleModel::getResetState(CState *state)
{
	CTransitionFunction::getResetState(state);
	state->setContinuousState(0, state->getContinuousState(0) * 0.8);
	state->setContinuousState(1, 0.0);
	state->setContinuousState(3, 0.0);
	state->setContinuousState(4, state->getContinuousState(2));
}

CCartPoleRewardFunction::CCartPoleRewardFunction(CCartPoleModel *model) : CStateReward(model->getStateProperties())
{
	this->cartpoleModel = model;
	useHeighPeak = true;
	punishOverRotate = true;
}


double CCartPoleRewardFunction::getStateReward(CState *state)
{
	double Phi = state->getContinuousState(2);
	double x = state->getContinuousState(0);

	double reward = cos(Phi) - 1 - 100 * my_exp((fabs(x) - cartpoleModel->lengthTrack / 2) * 25);

	if (useHeighPeak)
	{
		double dreward = exp(-pow(Phi, 2.0) * 25);
		reward += dreward;
	}
	if (punishOverRotate)
	{
		double phi_ = state->getContinuousState(4);
		double dreward = 20 * exp(fabs(phi_) - 10 * M_PI);
		reward -= dreward;
	}

	return reward;
}

void CCartPoleRewardFunction::getInputDerivation(CState *modelState, ColumnVector *targetState)
{
	double Phi = modelState->getState(properties)->getContinuousState(2);
	double x = modelState->getContinuousState(0);

	if (x < 0)
	{
		targetState->element(0) =  25 * 100 * exp((fabs(x) - cartpoleModel->lengthTrack / 2) * 25);
	}
	else
	{
		targetState->element(0) =  - 25 * 100 * exp((fabs(x) - cartpoleModel->lengthTrack / 2) * 25);
	}

	targetState->element(2) = 0.0;

	if (useHeighPeak)
	{
		
        targetState->element(2) = - 50 * Phi * exp( -pow(Phi, 2.0) * 25);
	}

	if (punishOverRotate)
	{
		double phi_ = modelState->getContinuousState(4);
		if (phi_ < 0)
		{
			targetState->element(4) = 20 * exp(fabs(phi_) - 10 * M_PI);
		}
		else
		{
			targetState->element(4) = - 20 * exp(fabs(phi_) - 10 * M_PI);
		}
	}

	targetState->element(1) = 0;
	targetState->element(2) = targetState->element(2) - sin(Phi);
	targetState->element(3) = 0;
}

CCartPoleHeightRewardFunction::CCartPoleHeightRewardFunction(CCartPoleModel *model) : CStateReward(model->getStateProperties())
{
	this->cartpoleModel = model;
}


double CCartPoleHeightRewardFunction::getStateReward(CState *state)
{
	double Phi = state->getContinuousState(2);
	//double x = state->getContinuousState(0);

	double reward = cos(Phi) - 1;
	return reward;
}

void CCartPoleHeightRewardFunction::getInputDerivation(CState *modelState, ColumnVector *targetState)
{
	double Phi = modelState->getState(properties)->getContinuousState(2);
//	double x = modelState->getContinuousState(0);

	targetState->element(1) = 0;
	targetState->element(2) = - sin(Phi);
	targetState->element(3) = 0;
}

#ifdef RL_TOOLBOX_USE_QT

CQTCartPoleVisualizer::CQTCartPoleVisualizer(CCartPoleModel *cartModel, QWidget *parent, const char *name) : CQTModelVisualizer(NULL, name)
{
	this->cartModel = cartModel;

	phi = 0;
	dphi = 0;

	x = 0;
	dx = 0;

	setFixedSize(700, 400);
}

void CQTCartPoleVisualizer::doDrawState( QPainter *painter)
{
	QString s1 = "x = " + QString::number( x );
	QString s2 = "x' = " + QString::number( dx );

	QString s3 = "Phi = " + QString::number( phi );
	QString s4 = "Phi' = " + QString::number( dphi );

	painter->drawText(10,20, s1);
	painter->drawText(10,40, s2);

	painter->drawText(10,60, s3);
	painter->drawText(10,80, s4);


	painter->translate(this->width() / 2, this->height() / 2);
	painter->setBrush(black);

	painter->drawRect(- cartModel->lengthTrack / 2 * 100 - 60, -25, 5, 50);
	painter->drawRect(cartModel->lengthTrack / 2 * 100 + 50, -25, 5, 50);

	painter->translate(x * 100, 0);

	painter->setBrush(white);
	painter->drawRect(-50, -25, 100, 50);

	painter->rotate(- phi + 180);

	painter->setBrush(black);
	painter->drawRect(- 5, -5 , 5, (cartModel->lengthPole * 100) + 5);

	painter->flush();
}

void CQTCartPoleVisualizer::newDrawState(CStateCollection *state)
{
	x = state->getState()->getContinuousState(0);
	dx = state->getState()->getContinuousState(1);

	phi = state->getState()->getContinuousState(2) * 180 / M_PI;
	dphi = state->getState()->getContinuousState(3) * 180 / M_PI;
}

#endif
