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

#include <math.h>

#include "ril_debug.h"
#include "cresiduals.h"
#include "cfeaturefunction.h"

CDiscreteResidual::CDiscreteResidual(double gamma)
{
	addParameter("DiscountFactor", gamma);
}

double CDiscreteResidual::getResidual(double oldV, double reward, double duration, double newV)
{
	double error = reward + pow(getParameter("DiscountFactor"), duration) * newV - oldV;
	//DebugPrint('r', "Resiudal: %f + %f * %f - %f = %f\n", reward, pow(getParameter("DiscountFactor"), duration), newV, oldV, error);
		   
	return error;
}


void CDiscreteResidual::getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *gradientResidualFeatures)
{
	gradientResidualFeatures->add(oldGradient, -1.0);
	gradientResidualFeatures->add(newGradient, pow(getParameter("DiscountFactor"), duration));
}

CContinuousEulerResidual::CContinuousEulerResidual(double dt, double sgamma)
{
	addParameter("TimeIntervall", dt);
	addParameter("ContinuousDiscountFactor", sgamma);
}


double CContinuousEulerResidual::getResidual(double oldV, double reward, double duration, double newV)
{
	double dt = getParameter("TimeIntervall");
	return (reward + (1 / (dt * duration) - getParameter("ContinuousDiscountFactor")) * newV - oldV / (dt * duration)) * dt;
}

void CContinuousEulerResidual::getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *gradientResidualFeatures)
{
	double dt = getParameter("TimeIntervall");

	gradientResidualFeatures->add(oldGradient, - 1 );
	gradientResidualFeatures->add(newGradient, (1  -  getParameter("ContinuousDiscountFactor") * (dt * duration)));
}

CContinuousCoulomResidual::CContinuousCoulomResidual(double dt, double sgamma) 
{
	addParameter("TimeIntervall", dt);
	addParameter("ContinuousDiscountFactor", sgamma);
}

double CContinuousCoulomResidual::getResidual(double oldV, double reward, double duration, double newV)
{
	double dt = getParameter("TimeIntervall");

	return (reward + (1 / (dt * duration) - getParameter("ContinuousDiscountFactor") / 2) * newV - oldV *(1 / (dt * duration) + getParameter("ContinuousDiscountFactor") / 2)) * dt;
}

void CContinuousCoulomResidual::getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *gradientResidualFeatures)
{
	double dt = getParameter("TimeIntervall");

	gradientResidualFeatures->add(oldGradient, (- 1  - getParameter("ContinuousDiscountFactor") * (dt * duration) / 2));
	gradientResidualFeatures->add(newGradient, (1 - getParameter("ContinuousDiscountFactor") *( dt * duration) / 2));
}

CConstantBetaCalculator::CConstantBetaCalculator(double beta)
{
	addParameter("ResidualBeta", beta);
}

double CConstantBetaCalculator::getBeta(CFeatureList *, CFeatureList *)
{
	return getParameter("ResidualBeta");
}


CVariableBetaCalculator::CVariableBetaCalculator(double mu, double maxBeta)
{
	addParameter("ResidualBetaMu", mu);
	addParameter("ResidualMaxBeta", maxBeta);
}

double CVariableBetaCalculator::getBeta(CFeatureList *directGradient, CFeatureList *residualGradient)
{
	

	double numerator = 0.0;
	double denominator = 0.0;

	double beta = 0.0;

	if (DebugIsEnabled('v'))
	{
		DebugPrint('v', "ResidualGradient: ");
		residualGradient->saveASCII(DebugGetFileHandle('v'));
		DebugPrint('v', "DirectGradient: ");
		directGradient->saveASCII(DebugGetFileHandle('v'));
	}

	numerator = residualGradient->multFeatureList(directGradient);
	denominator = residualGradient->multFeatureList(residualGradient) - numerator;


	if (fabs(denominator) < 0.0000001)
	{
		beta = 0.0;
	}
	else
	{
		beta = - numerator / denominator + getParameter("ResidualBetaMu");
	}

	DebugPrint('v', "Residual %f, %f: Beta = %f", numerator, denominator, beta);

	if (beta < 0.0 || beta > 1.0)
	{
		beta = 0.0;
	}
	else
	{
		if (beta > getParameter("ResidualMaxBeta"))
		{
			beta = getParameter("ResidualMaxBeta");
		}
	}
	DebugPrint('v', "beschränkt: %f\n", beta);


	return beta;
}

CResidualBetaFunction::CResidualBetaFunction(CAbstractBetaCalculator *betaCalculator, CResidualGradientFunction *residualGradient)
{
	this->betaCalculator = betaCalculator;
	this->residualGradient = residualGradient;

	addParameters(betaCalculator);
	addParameters(residualGradient);
}

void CResidualBetaFunction::getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *residualGradientFeatures)
{
	residualGradient->getResidualGradient(oldGradient, newGradient, duration, residualGradientFeatures);

	double beta = betaCalculator->getBeta(oldGradient, residualGradientFeatures);
	residualGradientFeatures->multFactor(beta);
	residualGradientFeatures->add(oldGradient, -(1 - beta));
}

void CDirectGradient::getResidualGradient(CFeatureList *oldGradient, CFeatureList *, double , CFeatureList *residualGradient)
{
	residualGradient->add(oldGradient, -1);
}

