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

#include "cacrobotmodel.h"
#include <math.h>

#include "cstateproperties.h"
#include "cstate.h"
#include "caction.h"
#include "ccontinuousactions.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CAcroBotModel::CAcroBotModel(double dt, double uMax, double length1, double length2, double mass1, double mass2, double mu_1, double mu_2, double g) : CLinearActionContinuousTimeTransitionFunction(new CStateProperties(4, 0),new CContinuousAction(new CContinuousActionProperties(1)), dt)
{
	this->uMax = uMax;
	this->length1 = length1;
	this->length2 = length2;
	this->mass1 = mass1;
	this->mass2 = mass2;
	this->mu_1 = mu_1;
	this->mu_2 = mu_2;
	this->g = g;

	properties->setMaxValue(0, M_PI);
	properties->setMinValue(0, -M_PI);

	properties->setPeriodicity(0, true);

	properties->setMaxValue(1, M_PI * 2);
	properties->setMinValue(1, -M_PI * 2);


	properties->setMaxValue(2, M_PI);
	properties->setMinValue(2, -M_PI);

	properties->setPeriodicity(2, true);

	properties->setMaxValue(3, M_PI * 3);
	properties->setMinValue(3, -M_PI * 3);

	contAction->getContinuousActionProperties()->setMaxActionValue(0,uMax);
	contAction->getContinuousActionProperties()->setMinActionValue(0,-uMax);


}

CAcroBotModel::~CAcroBotModel()
{
	delete properties;
	delete actionProp;
	delete contAction;
}

Matrix *CAcroBotModel::getB(CState *state)
{
	double Phi1 = state->getContinuousState(0);
//	double dPhi1 = state->getContinuousState(1);
	double Phi2 = state->getContinuousState(2) + Phi1;
//	double dPhi2 = state->getContinuousState(3);

	double denum = 16 * pow(length2/length1, 2) * (mass1 * mass2 / 9 + pow(mass2, 2) / 3)
		- 4 * pow(length1 * length2*mass2*cos(Phi1-Phi2), 2);

	double dPhi1_U = 4/3* pow(length2, 2) * mass2 * (-1) - 2 * length1 * length2 * mass2 * cos(Phi1 - Phi2);
	double dPhi2_U = 2 * length1 * length2 * mass2 * cos(Phi1 - Phi2) + 4*(mass1/3 + mass2) / pow(length1, 2);


	*B = 0.0;
	B->element(1, 0) = dPhi1_U / denum;
	B->element(3, 0) = dPhi2_U / denum;

	return B;	
}

ColumnVector *CAcroBotModel::getA(CState *state)
{
	double Phi1 = state->getContinuousState(0);
	double dPhi1 = state->getContinuousState(1);
	double Phi2 = state->getContinuousState(2) + Phi1;
	double dPhi2 = state->getContinuousState(3);

	double denum = 16 * pow(length2/length1, 2) * (mass1 * mass2 / 9 + pow(mass2, 2) / 3)
		- 4 * pow(length1 * length2*mass2*cos(Phi1-Phi2), 2);

	double b1, b2;

	b1 = 2 * mass2 * length2 * length1 * pow(dPhi2,2) * sin(Phi2-Phi1) + (mass1  + 2 * mass2) * sin(Phi1) * length1* g - mu_1 * dPhi1;
	b2 = 2 * mass2 * length2 * length1 * pow(dPhi1,2) * sin(Phi1-Phi2) + mass2 * sin(Phi2) * length2 * g - mu_2 * dPhi2;


	double ddPhi1 = 4/3* pow(length2, 2) * mass2 * b1 - 2 * length1 * length2 * mass2 * cos(Phi1 - Phi2) * b2;
	double ddPhi2 = - 2 * length1 * length2 * mass2 * cos(Phi1 - Phi2) * b1 + 4*(mass1/3 + mass2) / pow(length1, 2) * b2;

	ddPhi1 = ddPhi1 / denum;
	ddPhi2 = ddPhi2 / denum;

	A->element(0) = dPhi1;
	A->element(1) = ddPhi1;
	A->element(2) = dPhi2;
	A->element(3) = ddPhi2;

	return A;
}

bool CAcroBotModel::isFailedState(CState *)
{
	return false;
}

void CAcroBotModel::doSimulationStep(CState *state, double timestep, CAction *action, CActionData *data)
{
	getDerivationX(state, action, derivation, data);

	double ddPhi1 = derivation->element(1);
	double ddPhi2 = derivation->element(3);

	for (unsigned int i = 0; i < state->getNumContinuousStates(); i++)
	{
		state->setContinuousState(i, state->getContinuousState(i) + timestep * derivation->element(i));
	}

	state->setContinuousState(0, state->getContinuousState(0) + pow(timestep, 2) * ddPhi1 / 2);
	state->setContinuousState(2, state->getContinuousState(2) + pow(timestep, 2) * ddPhi2 / 2);
}

void CAcroBotModel::getResetState(CState *state)
{
	CTransitionFunction::getResetState(state);
	state->setContinuousState(1, 0.0);
	state->setContinuousState(3, 0.0);
//	state->setContinuousState(2, state->getContinuousState(2) - state->getContinuousState(0));
}

/*
CAcroBotModelSutton::CAcroBotModelSutton(double dt, double uMax, double length1, double length2, double mass1, double mass2, double I1, double I2, double g) : CAcroBotModel(dt, uMax, length1, length2, mass1, mass2, 0.05, 0.05, g)
{
	this->I1 = I1;
	this->I2 = I2;
}


CAcroBotModelSutton::~CAcroBotModelSutton()
{

}

Matrix *CAcroBotModelSutton::getB(CState *state)
{
//	double Phi1 = state->getContinuousState(0) + M_PI;
	double Phi2 = state->getContinuousState(2);

	double d1 = mass1 * pow(length1 / 2.0, 2) + mass2 * (pow(length1, 2) + pow(length2 / 2.0, 2) + length1 * length2 * cos(Phi2)) + I1 + I2;
	double d2 = mass2 * (pow(length2 / 2.0, 2) + length1 * length2 / 2.0 * cos(Phi2)) + I2;

	double dPhi2_U = 1 / ( mass2 * pow(length2 / 2.0, 2) + I2 - pow(d2,2) / d1);


	double dPhi1_U = - d2 / d1 * dPhi2_U;
	

	*B = 0.0;
	B->setElement(1, 0, dPhi1_U);
	B->setElement(3, 0, dPhi2_U);

	return B;	
	
}


ColumnVector *CAcroBotModelSutton::getA(CState *state)
{
	double Phi1 = state->getContinuousState(0) - M_PI;
	double Phi2 = state->getContinuousState(2);
	double dPhi1 = state->getContinuousState(1);
	double dPhi2 = state->getContinuousState(3);


	double d1 = mass1 * pow(length1/ 2.0, 2) + mass2 * (pow(length1, 2) + pow(length2 / 2.0, 2) + length1 * length2 * cos(Phi2)) + I1 + I2;
	double d2 = mass2 * (pow(length2 / 2.0, 2) + length1 * length2 / 2.0 * cos(Phi2)) + I2;

	double f2 = mass2 * length2 / 2.0 * g * cos(Phi1 + Phi2 - M_PI / 2.0);
	double f1 = - mass2 * length1 * length2 / 2.0 * pow(dPhi2, 2) * sin(Phi2) - 2 * mass2 * length1 * length2 / 2.0 * dPhi1 * dPhi2 * sin(Phi2);
	f1 += (mass1 * length1 / 2.0 + mass2 * length1) * g * cos(Phi1 - M_PI / 2) + f2;

	double ddPhi2 = 1 / ( mass2 * pow(length2 / 2.0, 2) + I2 - pow(d2,2) / d1) * (d2 / d1 * f1 - mass2 * length1 * length2 / 2.0 * pow(dPhi1, 2.0) * sin(Phi2) - f2);


	double ddPhi1 = - (d2 * ddPhi2 + f1) / d1;

	A->setElement(0, dPhi1);
	A->setElement(1, ddPhi1);
	A->setElement(2, dPhi2);
	A->setElement(3, ddPhi2);

	return A;
}
*/


CAcroBotRewardFunction::CAcroBotRewardFunction(CAcroBotModel *model, double segmentFactor) : CStateReward(model->getStateProperties())
{
	this->model = model;
	useHeighPeak = true;
	this->segmentFactor = segmentFactor;
	power = 1.0;

}

double CAcroBotRewardFunction::getStateReward(CState *state)
{

	double Phi1 = state->getContinuousState(0);
	double Phi2 = state->getContinuousState(2);

	double height = 2 *(segmentFactor * model->length1 * (cos(Phi1) + 1) + (1 - segmentFactor) * model->length2 * ( cos(Phi2 + Phi1) + 1)) / (model->length1 + model->length2) / 2.0;


	double reward = 2 * (pow(height, power) - 1.0);
	if (useHeighPeak)
	{
		double dreward = 0.5 * exp((-pow(Phi1, 2.0) - pow(Phi2, 2.0)) * 25);
		reward += dreward;
	}

	return reward;
}

void CAcroBotRewardFunction::getInputDerivation(CState *modelState, ColumnVector *targetState)
{
	double Phi1 = modelState->getState(properties)->getContinuousState(0);
	double Phi2 = modelState->getState(properties)->getContinuousState(2);

	targetState->element(0) = - segmentFactor * model->length1 * sin(Phi1);
	targetState->element(1) = 0;
	targetState->element(2) = -(1 - segmentFactor) * model->length2 * sin(Phi2 + Phi1);
	targetState->element(3) = 0;

	if (useHeighPeak)
	{
		targetState->element(0) = targetState->element(0) - 25 * (Phi1) * exp( (-pow(Phi1, 2.0) - pow(Phi2, 2.0)) * 25);
		targetState->element(2) = targetState->element(2) - 25 * (Phi1) * exp( (-pow(Phi1, 2.0) - pow(Phi2, 2.0)) * 25);
	}
}

CAcroBotHeightRewardFunction::CAcroBotHeightRewardFunction(CAcroBotModel *model) : CStateReward(model->getStateProperties())
{
	this->model = model;
	useHeighPeak = true;
}

double CAcroBotHeightRewardFunction::getStateReward(CState *state)
{
	double Phi1 = state->getContinuousState(0);
	double Phi2 = state->getContinuousState(2);

	double reward = model->length1 * (cos(Phi1) - 1) + model->length2 * ( cos(Phi2 + Phi1) - 1);

	return reward;
}

void CAcroBotHeightRewardFunction::getInputDerivation(CState *modelState, ColumnVector *targetState)
{
	double Phi1 = modelState->getState(properties)->getContinuousState(0);
	double Phi2 = modelState->getState(properties)->getContinuousState(2);

	targetState->element(0) = - model->length1 * sin(Phi1);
	targetState->element(1) = 0;
	targetState->element(2) = - model->length2 * sin(Phi2 + Phi1);
	targetState->element(3) = 0;
}

CAcroBotExpRewardFunction::CAcroBotExpRewardFunction(CAcroBotModel *l_model, double l_expFactor) : CStateReward(l_model->getStateProperties())
{
	this->model = l_model;
	this->expFactor = l_expFactor;
}

double CAcroBotExpRewardFunction::getStateReward(CState *state)
{
	double Phi1 = state->getContinuousState(0);
	double Phi2 = state->getContinuousState(2);

	double reward = model->length1 * (cos(Phi1) - 1) + model->length2 * ( cos(Phi2 + Phi1) - 1);

	reward = -1.0 + exp(expFactor * reward);

	return reward;
}

void CAcroBotExpRewardFunction::getInputDerivation(CState *, ColumnVector *)
{

}

CAcroBotVelocityRewardFunction::CAcroBotVelocityRewardFunction(CAcroBotModel *model) :  CStateReward(model->getStateProperties())
{
	this->model = model;
	invertVelocity = false;
}

double CAcroBotVelocityRewardFunction::getStateReward(CState *state)
{
	double reward = fabs(state->getContinuousState(1));

	if (! invertVelocity)
	{
		reward = - 1.0 + reward; 
	}
	else
	{
		reward = - reward;
	}
	return reward;
}

void CAcroBotVelocityRewardFunction::getInputDerivation(CState *, ColumnVector *)
{

}



#ifdef RL_TOOLBOX_USE_QT

CQTAcroBotVisualizer::CQTAcroBotVisualizer( CAcroBotModel *acroModel, QWidget *parent, const char *name) : CQTModelVisualizer(parent, name)
{

	this->acroModel = acroModel;

	phi1 = 0;
	dphi1 = 0;

	phi2 = 0;
	dphi2 = 0;
}

void CQTAcroBotVisualizer::doDrawState( QPainter *painter)
{
	QString s1 = "Phi1 = " + QString::number( phi1 );
	QString s2 = "Phi1' = " + QString::number( dphi1 );

	QString s3 = "Phi2 = " + QString::number( phi2 );
	QString s4 = "Phi2' = " + QString::number( dphi2 );

	painter->drawText(10,20, s1);
	painter->drawText(10,40, s2);

	painter->drawText(10,60, s3);
	painter->drawText(10,80, s4);


	painter->translate(this->width() / 2, this->height() / 2);

	painter->rotate(- phi1 + 180);

	painter->drawRect(- 5, -5, 10, (acroModel->length1 * 100) + 10);

	painter->translate(0, acroModel->length1 * 100);

	painter->rotate(- phi2);

	painter->drawRect(- 5, -5, 10, (acroModel->length2 * 100) + 10);

	painter->flush();
}

void CQTAcroBotVisualizer::newDrawState(CStateCollection *state)
{
	phi1 = state->getState()->getContinuousState(0) * 180 / M_PI;
	dphi1 = state->getState()->getContinuousState(1) * 180 / M_PI;

	phi2 = state->getState()->getContinuousState(2) * 180 / M_PI;
	dphi2 = state->getState()->getContinuousState(3) * 180 / M_PI;
}

#endif
