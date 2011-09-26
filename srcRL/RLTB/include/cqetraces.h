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

#ifndef C_ETRACES_H
#define C_ETRACES_H

#include "cparameters.h"

class CAbstractQFunction;
class CQFunction;
class CComposedQFunction;
class CGradientQFunction;

class CAction;
class CActionData;
class CStateCollection;
class CAbstractVETraces;

class CFeatureList;
/// Interface for Q-ETraces
/** Q-ETraces store additionally the action to the state, so
you can trace back the episode an make updates to past states. The class provides functions for reseting, updating and add Q-ETraces.
In contains the attenuation factor lambda as Parameter "Lambda", which is used to attenuate E-Traces from the past.
You can also set, wether the ETraces should be replacing ETraces or not (Parameter "ReplacingETraces"). This parameter is handled by the subclasses slight differently, for more details see the subclasses. 
<p>
CAbstractQETraces has following parameters:
- "Lambda", 0.9 : attenuation factor
- "DiscountFactor", 0.95 : gamma
- "ReplacingETraces", true
*/

class CAbstractQETraces : virtual public CParameterObject 
{
protected:
/// The assigned Q-Function for updating
	CAbstractQFunction *qFunction;

public:
	CAbstractQETraces(CAbstractQFunction *qFunction);
	virtual ~CAbstractQETraces() {};

/// Interface function for reseting the ETraces
	virtual void resetETraces() = 0;
/// Interface function for updating the ETraces
/**
I.e. all stored ETraces-factors get multiplied by lambda * gamma, gamma is taken from the Q-Function. 
If the action is a multistep action, the factor lambda * gamma is exponentiated with the duration.
*/
	virtual void updateETraces(CAction *action, CActionData *data = NULL) = 0;
/// Interface function for adding a State-Action pair with the given factor to the ETraces
	virtual void addETrace(CStateCollection *State, CAction *action, double factor = 1.0, CActionData *data = NULL) = 0;

/// Interface function for updating the Q-Values of all State-Action Pairs in the ETraces
	virtual void updateQFunction(double td) = 0;
 
	/// sets the Parameter "Lambda"
	virtual void setLambda(double lambda);
	/// returns the Parameter "Lambda"
	virtual double getLambda();

	/// sets the parameter "ReplacingETraces"
	virtual void setReplacingETraces(bool bReplace);
	/// returns the parameter "ReplacingETraces"
	virtual bool getReplacingETraces();
};

/// Q-ETraces consisting of V-Etraces (one for each action)
/**Q-ETraces have the same structure of their corresponding Q-Function. The class CQETraces maintains a list
of V-ETraces, for each V-Function of the assigned Q-Function an own ETrace. The list gets initialized with the standard ETraces for the specific V-Function (see getStandardETraces), but they can also be set explicitly. When
a ETrace is added, all V-ETraces get updated (the factors are multiplied by lambda * gamma(discountfactor) and the state collection is added to the VETrace object for the current action.
When the resetEtraces and updateQFunction functions are called from the CQETraces class, the similar functions from all V-ETraces are called.
CQETraces has following parameters:
see CAbstractQETraces
*/

class CQETraces : public CAbstractQETraces
{
protected:
	/// The list of V-ETraces
	std::list<CAbstractVETraces *> *vETraces;
	/// the assigned composed Q-Function
	CQFunction *qExFunction;

public:
	/// Creates a composed Q-ETraces Object.
/**
The V-ETraces are comming from the getStandardVETraces function of the coresponding V-ETraces from the compounded Q-Function object.
*/
	CQETraces(CQFunction *qfunction);
	virtual ~CQETraces();

/// Resets all V-ETraces	
	virtual void resetETraces();
/// Adds the state to the V-Etrace which is assigned to the specified action with the given factor.
	virtual void addETrace(CStateCollection *State, CAction *action, double factor = 1.0, CActionData *data = NULL);
/// Calls the updateETraces function from all V-ETraces
/**
So all ETraces factors get multplied by lambda * gamma. For an multistep action the update has to be 
lambda * gamma^N, so the action is given as parameter to determine the duration.<p>
The function calls the updateETraces function of all V-ETraces with the duration of the action as parameter.
*/
	virtual void updateETraces(CAction *action, CActionData *data = NULL);

/// Calls the updateVFunction method from th CAbstractVETraces class. 
/**
So all state-action Pairs get updated by td * ETraceFaktor
*/
	virtual void updateQFunction(double td);

/// Sets an ETrace Object for a given index
/**
it is recommended to use the standard ETraces. This function gives you the possibility to 
use oter ETraces objects for you V-Function with index "index".
<p>
If bDeleteOld is true, the old VETraces object gets deleted.
*/
	void setVETrace(CAbstractVETraces *vEtrace, int index, bool bDeleteOld = true);
/// Returns the index th VETraces object
	CAbstractVETraces *getVETrace(int index);

/// sets the internal replacing ETraces flag and the flag of the V-ETraces
	virtual void setReplacingETraces(bool bReplace);
};

/// This is the E-Trace class for the CComposedQFunction class
/** 
This class represents the ETrace object for the composed Q-Functions. Composed Q-Functions consists of several Q-Functions, so you can use different kind of Q-Functions for different actions (For example combine a NeuralNetwork Q-Function with a Feature Q-Function). The class works similar to CQETraces, but instead of CVETrace objects, it handles CQETrace objects.
*/
class CComposedQETraces : public CAbstractQETraces
{
protected:
	/// The list of Q-ETraces
	std::list<CAbstractQETraces *> *qETraces;
	/// the assigned composed Q-Function
	CComposedQFunction *qCompFunction;

public:

	CComposedQETraces(CComposedQFunction *qfunction);
	virtual ~CComposedQETraces();


	virtual void resetETraces();
	virtual void addETrace(CStateCollection *State, CAction *action, double factor = 1.0, CActionData *data = NULL);
	
	virtual void updateETraces(CAction *action, CActionData *data = NULL);
	virtual void updateQFunction(double td);

	void setQETrace(CAbstractQETraces *qEtrace, int index, bool bDeleteOld = true);

	/// Returns the index th VETraces object
	CAbstractQETraces *getQETrace(int index);

	/// sets the internal replacing ETraces flag and the flag of the Q-ETraces
	virtual void setReplacingETraces(bool bReplace);
};



/// E-Trace class for all gradient Q-Functions
/** 
This class mantains the etraces for gradient q-Functions. For the ETraces it uses a feature list, every time an etrace is added with addETrace the e-trace object calculates the gradient with respect to the weights from the current state of the Q-Function and adds this gradient to the e-trace feature list. For gradient E-Traces there are 2 different ways of adding a gradient, wether you want to have replacing or non-replacing etraces:
- Non-Replacing ETraces : The gradient is just added to the current ETraces.
- Replacing ETraces : If the current gradient has the same sign as the current ETrace, the greater value of both remains the new E-Trace value. If the signs are different, the current gradient's value is taken.
For updating the gradient Q-Function the E-Trace object naturally uses the CGradientUpdateFunction interface of the Q-Function.
<p>
CGradientQETraces has the following Parameters:
-"Lambda", 0.9 : attenuation factor
-"DiscountFactor", 0.95 : gamma
-"ReplacingETraces", see description for adding a gradient to the etrace list.
-"ETraceTreshold", 0.001 : smallest value of an etrace, the etrace will be deleted from the list if its lower than this value. Used for performance reasons.
-"ETraceMaxListSize", 100 : Maximum size of the etrace-list, if the list exceeds this value, the smalles etraces will be deleted. 

*/

class CGradientQETraces : public CAbstractQETraces
{
protected:
	CGradientQFunction *gradientQFunction;
	CFeatureList *eTrace;
	CFeatureList *gradient;

public:

	CGradientQETraces(CGradientQFunction *qfunction);
	virtual ~CGradientQETraces();

/// Resets the etrace object, clears the etrace feature list.
	virtual void resetETraces();

	/// Calls addGradientETrace with the current gradient
	virtual void addETrace(CStateCollection *State, CAction *action, double factor = 1.0, CActionData *data = NULL);

	/// Adds the current gradient to the etrace list
	/** Every time an etrace is added with addETrace the e-trace object calculates the gradient with respect to the weights from the current state of the Q-Function and adds this gradient to the e-trace feature list. For gradient E-Traces there are 2 different ways of adding a gradient, wether you want to have replacing or non-replacing etraces:
	- Non-Replacing ETraces : The gradient is just added to the current ETraces.
	- Replacing ETraces : If the current gradient has the same sign as the current ETrace, the greater value of both remains the new E-Trace value. If the signs are different, the gradient's values are added.	*/
	virtual void addGradientETrace(CFeatureList *gradient, double factor = 1.0);

	/// Multiplies as usaul all etraces with lambda  * gamma
	virtual void updateETraces(CAction *action, CActionData *data = NULL);
	/// Updates the QFunction with its CGradientUpdateFunction interface
	virtual void updateQFunction(double td);
	
	/// returns the current gradient list.
	CFeatureList *getGradientETraces() {return eTrace;};
};

#endif

