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

#include "cpendulummodel.h"
#include "cstateproperties.h"
#include "cstate.h"
#include "cstatecollection.h"
#include "caction.h"
#include "ccontinuousactions.h"
#include <math.h>

CPendulumModel::CPendulumModel(double dt, double uMax , double dPhiMax, double length, double mass, double mu, double g) : CLinearActionContinuousTimeTransitionFunction(new CStateProperties(2,0), new CContinuousAction(new CContinuousActionProperties(1)), dt)
{
	this->uMax = uMax;
	this->dPhiMax = dPhiMax;
	this->length = length;
	this->mass = mass;
	this->mu = mu;
	this->g = g;

	/*addParameter("UMax", uMax);
	addParameter("DPhiMax",dPhiMax);
	addParameter("Length", length);
	addParameter("Mass", mass);
	addParameter("Friction", mu);
	addParameter("Gravity", g);*/

	actionProp->setMaxActionValue(0, uMax);
	actionProp->setMinActionValue(0, -uMax);

	properties->setMaxValue(0, M_PI);
	properties->setMinValue(0, - M_PI);

	properties->setPeriodicity(0, true);

	properties->setMaxValue(1, dPhiMax);
	properties->setMinValue(1, - dPhiMax);

}

CPendulumModel::~CPendulumModel()
{
	delete properties;
	delete actionProp;
	delete contAction;
}

Matrix *CPendulumModel::getB(CState *)
{
/*	double mass = getParameter("Mass");
	double length = getParameter("Length");*/
	B->element(0,0) = 0.0;
	B->element(1,0) = 1 / (mass * pow(length,2));

	return B;
}

ColumnVector *CPendulumModel::getA(CState *state)
{
	/*double mass = getParameter("Mass");
	double length = getParameter("Length");
	double mu = getParameter("Friction");
	double g = getParameter("Gravity");
*/
	double dphi = state->getContinuousState(1);
	A->element(0) = dphi;
	double ddphi = 1 / (mass * pow(length,2)) * (- mu * state->getContinuousState(1) + mass * g * length * sin(state->getContinuousState(0)));
	A->element(1) = ddphi;

	return A;
}

void CPendulumModel::setParameter(string paramName, double value)
{
	if (paramName == "UMax")
	{
		actionProp->setMaxActionValue(0, value);
		actionProp->setMinActionValue(0, -value);
	}
	if (paramName == "DPhiMax")
	{
		properties->setMaxValue(1, value);
		properties->setMinValue(1, - value);
	}
	CParameterObject::setParameter(paramName, value);
}


bool CPendulumModel::isFailedState(CState *)
{
	return false;
}

void CPendulumModel::getResetState(CState *resetState)
{
	CTransitionFunction::getResetState(resetState);
	
	if (resetType != DM_RESET_TYPE_ALL_RANDOM)
	{
		resetState->setContinuousState(1, 0);
	}
}



void CPendulumModel::doSimulationStep(CState *state, double timestep, CAction *action, CActionData *data)
{
	getDerivationX(state, action, derivation, data);

	double ddPhi = derivation->element(1);

	state->setContinuousState(0, state->getContinuousState(0) + timestep * derivation->element(0) + pow(timestep,2) / 2 * ddPhi);
	state->setContinuousState(1, state->getContinuousState(1) + timestep * derivation->element(1));
}

CPendulumRewardFunction::CPendulumRewardFunction(CPendulumModel *model) : CStateReward(model->getStateProperties())
{
	rewardFactor = 1.0;
}


double CPendulumRewardFunction::getStateReward(CState *state)
{
	double Phi = state->getContinuousState(0);
	return rewardFactor * (cos(Phi) - 1);
}

void CPendulumRewardFunction::getInputDerivation(CState *modelState, ColumnVector *targetState)
{
	double Phi = modelState->getState(properties)->getContinuousState(0);
	targetState->element(1) = 0;
	targetState->element(0) = - sin(Phi);
}


#ifdef RL_TOOLBOX_USE_QT
CQTPendulumVisualizer::CQTPendulumVisualizer( CPendulumModel *pendModel, QWidget *parent, const char *name) : CQTModelVisualizer(parent, name)
{
	this->pendModel = pendModel;
	phi = 0;
	dphi = 0;

	this->setCaption("Pendulum Model Visualizer");
}

void CQTPendulumVisualizer::doDrawState( QPainter *painter)
{
	//QString s1 = "Phi = " + QString::number( phi );
	//QString s2 = "Phi' = " + QString::number( dphi );

	//painter->drawText(10,20, s1);
	//painter->drawText(10,40, s2);

	painter->drawLine(0, drawWidget->height() / 2, drawWidget->width(), drawWidget->height() / 2);
	painter->translate(drawWidget->width() / 2, drawWidget->height() / 2);
	painter->rotate(phi + 180);

	painter->setBrush(black);
	painter->drawRect(- 3, -5, 6, (pendModel->length * 100) + 10);
	painter->translate(0,(pendModel->length * 100) + 5);
	painter->drawEllipse(-6,-6,12,12);

	painter->flush();
}


void CQTPendulumVisualizer::newDrawState(CStateCollection *state)
{
	phi = state->getState()->getContinuousState(0) * 180 / M_PI;
	dphi = state->getState()->getContinuousState(1) * 180 / M_PI;
}

#endif

CPendulumUpTimeCalculator::CPendulumUpTimeCalculator(double phi_up, double dt)
{
	this->phi_up = phi_up;
	this->dt = dt;

	this->up_steps = 0;
}

void CPendulumUpTimeCalculator::nextStep(CStateCollection *oldState, CAction *, CStateCollection *)
{
	if (fabs(oldState->getState()->getContinuousState(0)) < phi_up) 
	{
		up_steps ++;
	}
}

void CPendulumUpTimeCalculator::newEpisode()
{
	up_steps = 0;
}

double CPendulumUpTimeCalculator::getUpTime()
{
	return up_steps * dt;
}

int CPendulumUpTimeCalculator::getUpSteps()
{
	return up_steps;
}
/*
bool CTestSuitePendulumUpTimeCalculatorEvaluator::isEpisodeSuccessFull(FILE *stream)
{
	printf("Upsteps %d (needed %d)\n", upTimeCalc->getUpSteps(), neededUpSteps);

	if (stream)
	{
		fprintf(stream,"Upsteps %d\n", upTimeCalc->getUpSteps());
	}

	return upTimeCalc->getUpSteps() >= neededUpSteps;
}

CTestSuitePendulumUpTimeCalculatorEvaluator::CTestSuitePendulumUpTimeCalculatorEvaluator(CAgent *agent, int neededSuccEpisodes, int maxEpisodes, int stepsPerEpisode, int neededUpSteps, double phi_up) : CTestSuiteEpisodesToLearnEvaluator(agent, neededSuccEpisodes, maxEpisodes, stepsPerEpisode)
{
	this->neededUpSteps = neededUpSteps;

	upTimeCalc = new CPendulumUpTimeCalculator(phi_up, 1.0);

	agent->addSemiMDPListener(upTimeCalc);

}

CTestSuitePendulumUpTimeCalculatorEvaluator::~CTestSuitePendulumUpTimeCalculatorEvaluator()
{
	agent->removeSemiMDPListener(upTimeCalc);
	delete upTimeCalc;
}
*/

