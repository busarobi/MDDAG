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

#ifndef C_RESIDUALS__H
#define C_RESIDUALS__H


#include "cparameters.h"

class CFeatureList;
class CStateCollection;


class CResidualGradientFunction : virtual public CParameterObject
{
public:
	virtual void getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *residualGradientFeatures) = 0;
};

class CResidualFunction : public CResidualGradientFunction
{
public:
	virtual double getResidual(double oldV, double reward, double duration, double newV) = 0;
};


class CDiscreteResidual : public CResidualFunction
{
protected: 
public:
	CDiscreteResidual(double gamma);

	virtual double getResidual(double oldV, double reward, double duration, double newV);
	
	virtual void getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *residualGradientFeatures);
};

class CContinuousEulerResidual : public CResidualFunction
{
protected:

public:

	CContinuousEulerResidual(double dt, double sgamma);
	
	virtual double getResidual(double oldV, double reward, double duration,  double newV);

	virtual void getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *residualGradientFeatures);
};

class CContinuousCoulomResidual : public CResidualFunction
{
protected:
	
public:
	CContinuousCoulomResidual(double dt, double sgamma);

	virtual double getResidual(double oldV, double reward, double duration, double newV);

	virtual void getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *residualGradientFeatures);
};

class CAbstractBetaCalculator : virtual public CParameterObject
{
public:
	virtual double getBeta(CFeatureList *directGradient, CFeatureList *residualGradient) = 0;
};

class CConstantBetaCalculator : public CAbstractBetaCalculator
{
protected:
public:
	CConstantBetaCalculator(double beta);
	virtual double getBeta(CFeatureList *directGradient, CFeatureList *residualGradient);
};

class CVariableBetaCalculator : public CAbstractBetaCalculator
{
protected:

public:
	CVariableBetaCalculator(double mu, double maxBeta);
	virtual double getBeta(CFeatureList *directGradient, CFeatureList *residualGradient);
};


class CResidualBetaFunction : public CResidualGradientFunction
{
protected:
	CAbstractBetaCalculator *betaCalculator;
	CResidualGradientFunction *residualGradient;
	CFeatureList *tempResidual;
public:
	CResidualBetaFunction(CAbstractBetaCalculator *betaCalculator, CResidualGradientFunction *residualGradient);

	virtual void getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *residualGradientFeatures);

};

class CDirectGradient : public CResidualGradientFunction
{
public:
	virtual void getResidualGradient(CFeatureList *oldGradient, CFeatureList *newGradient, double duration, CFeatureList *residualGradientFeatures);

};




#endif

