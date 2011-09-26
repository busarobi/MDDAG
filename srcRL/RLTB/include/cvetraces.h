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

#ifndef C_VETRACES__H
#define C_VETRACES__H

#include "cparameters.h"
#include <list>

class CAbstractVFunction;
class CGradientVFunction;

class CState;
class CStateCollection;
class CFeatureVFunction;
class CFeatureList;
class CStateProperties;
class CStateCollectionList;
class CStateCollectionImpl;
class CStateModifier;

/// Class representing etraces for a V-Function
/** V-ETraces store the "Trace" of each state, so all past states are stored in that "Trace". Many Learning Algorithms use E-Traces for their Value-Updates, which makes a very good improvement to most of the algorithms.  V-ETraces objects  stores the E-Traces only for the states, not for actions (see CAbstractQETraces), what exactly gets stored in the etraces depends on the kind of the value function. The class CAbstractVETraces is the interface for the V-Etraces objects. It has 4 virtual functions for 
implementing the E-Traces functionality.
<ul>
<li> addETrace(CStateCollection *State, double factor = 1.0) adds the etrace of the specified state with the given factor. </li>
<li> updateVFunction(double td): updates the VFunction with the ETraces. For each state in the ETraces the update value "td" gets multiplied by the state's factor and then the V-Value of the specified state is updated by this value. </li>
<li> resetETraces(): resets the ETraces, i.e. all states are cleared from the ETraces. </li>
<li> updateETraces(int duration): Multiplies all ETraces with (lambda * gamma)^duration. The duration is obviously the duration of the current step, so the time attenuation factor has to be exponentiated with the duration. </li>
</ul>
Each ETraces object has the parameter "Lambda", which is a attentuation factor. Every state from the past gets updated by
updateVFunction with the factor lambda^N*gamma^N, N is the time past since the state was active the parameter gamma is the discount factor of the learning Problem (Parameter: "DiscountFactor").
\par
In The RIL toolbox there are several implementations of V-ETraces. CStateVETraces stores the states directly in a state list and maintains an own list for the factors, CFeatureVETraces saves only the features of a state in a Table and CGradientVETraces saves always the current gradient to the etrace object. To determine which E-Trace object shall be used for a value-function the class CAbstractVFunction provides the
method getStandardETraces, which returns a new VETraces object which is best suited for that class of V-Functions.
<p>
The class CAbstractVETraces has following parameters:
- "Lambda", 0.9 : attenuation factor
- "DiscountFactor", 0.95 : gamma
- "ReplacingETraces", replacing etrace handling depends on the kind of the etraces
- "ETraceTreshold", 0.001 : smallest value of an etrace, the etrace will be deleted from the list if its lower than this value. Used for performance reasons.

*/
class CAbstractVETraces : virtual public CParameterObject
{
protected:

///pointer to the V-Function
	CAbstractVFunction *vFunction;
/// Use replacing etraces? Used for feature ETraces
public:
/// Creates an ETrace for the given V-Function
	CAbstractVETraces(CAbstractVFunction *vFunction);

/// Interface for clearing the Etraces.
	virtual void resetETraces() = 0;
/// Interface for adding a Etrace
	virtual void addETrace(CStateCollection *State, double factor = 1.0) = 0;
/// Interfeace for updating the ETraces.
/**
All ETraces factors get multplied by lambda * gamma. For an multistep action the update has to be 
lambda * gamma^N, so the duration can be given as parameter.<p>
*/
	virtual void updateETraces(int duration = 1) = 0;
/// Update the V-Function
/** For all States in the Etraces the "td" value is multiplied with the E-Trace factor 
(e.g. lambda^N*gamma^N$ for replacing E-Traces), N is the time past since the state was active is calculated 
 ,the Value of the state is updated.
 */
	virtual void updateVFunction(double td) = 0;
 
	void setLambda(double lambda);
	double getLambda();
		
	void setTreshold(double treshold);
	double getTreshold();

/// Sets the use of Replacing V-Etraces. 
	void setReplacingETraces(bool bReplace);
	bool getReplacingETraces();

	CAbstractVFunction *getVFunction();
};

/// State ETraces stores the state object itself. 
/** Stores states of the kind the V-Function uses. For each state there is an Etrace factor stored in a double list,
which are updated when an ETrace is added (every factor gets multiplied by lambda^N * gamma^N), the active state is
intialised with the given factor (standard value is 1.0). 
<p>
State VETraces can be used for non-parametric Value Functions, when they also don't depend on a feature state. But they are very slow and its not recommended to use state etraces*/

class CStateVETraces :  public CAbstractVETraces
{
protected:
	CStateCollectionList *eTraceStates;
	int eTraceLength;
	CStateCollectionImpl *bufState;

	std::list<double> *eTraces;
public:
	
	CStateVETraces(CAbstractVFunction *vFunction, CStateProperties *modelState, std::list<CStateModifier *> *modifiers = NULL);

	virtual ~CStateVETraces();

	virtual void resetETraces();
	virtual void addETrace(CStateCollection *State, double factor = 1.0);
	virtual void updateETraces(int duration = 1);
	
	virtual void updateVFunction(double td);
};

/// ETraces for gradient functions
/**
This class mantains the etraces for gradient V-Functions. For the ETraces it uses a feature list, every time an etrace is added with addETrace the e-trace object calculates the gradient with respect to the weights from the current state of the Q-Function and adds this gradient to the e-trace feature list. For gradient E-Traces there are 2 different ways of adding a gradient, wether you want to have replacing or non-replacing etraces:
- Non-Replacing ETraces : The gradient is just added to the current ETraces.
- Replacing ETraces : If the current gradient has the same sign as the current ETrace, the greater value of both remains the new E-Trace value. If the signs are different, the current gradient's value is taken.
For updating the gradient Q-Function the E-Trace object naturally uses the CGradientUpdateFunction interface of the V-Function.
<p>
CGradientQETraces has the following Parameters:
-"Lambda", 0.9 : attenuation factor
-"DiscountFactor", 0.95 : gamma
-"ReplacingETraces", see description for adding a gradient to the etrace list.
-"ETraceTreshold", 0.001 : smallest value of an etrace, the etrace will be deleted from the list if its lower than this value. Used for performance reasons.
-"ETraceMaxListSize", 100 : Maximum size of the etrace-list, if the list exceeds this value, the smalles etraces will be deleted. 
*/
class CGradientVETraces : public CAbstractVETraces
{
protected:

/// list for the Etraces
	CFeatureList *eFeatures;
	CFeatureList *tmpList;

	CGradientVFunction *gradientVFunction;

public:
	CGradientVETraces(CGradientVFunction *gradientVFunction);

	virtual ~CGradientVETraces();

/// Clear the etrace-feature list
/**
All elements are deleted from the etraces list and added to the resource list.
*/
	virtual void resetETraces();
/// Adds the state to the etrace List
/**
Calls addGradientETrace
*/
	virtual void addETrace(CStateCollection *State, double factor = 1.0);
/// Updates all etrace factors
/**
All etrace factors get multiplied by lambda * gamma^N, where the duration parameter is N.
*/
	virtual void updateETraces(int duration = 1);

	virtual void multETraces(double factor);
	
/// All discrete states in the etrace list gets updated
/**
The update of the V-Function for state s_i is td * e_i, where e_i is the etrace factor of s_i.
*/
	virtual void updateVFunction(double td);

	// Adds the current gradient to the etrace list
	/** Every time an etrace is added with addETrace the e-trace object calculates the gradient with respect to the weights from the current state of the V-Function and adds this gradient to the e-trace feature list. For gradient E-Traces there are 2 different ways of adding a gradient, wether you want to have replacing or non-replacing etraces:
	- Non-Replacing ETraces : The gradient is just added to the current ETraces.
	- Replacing ETraces : If the current gradient has the same sign as the current ETrace, the greater value of both remains the new E-Trace value. If the signs are different, the gradients are added.	*/
	virtual void addGradientETrace(CFeatureList *gradient, double factor);

	CFeatureList* getGradientETraces();

};

class CFeatureVFunction;


/// This class is used as ETraces for feature V-Functions
/** 
The class has the same functionality as the gradient v-etraces class, the difference with features is that the gradient here is already calculated by the state modifiers (since feature V-Functions are linear approximators), so the modified feature state is added directly to the etrace list (slightly better performance):
The class has the following parameters:
Same as CGradientVETraces
*/
class CFeatureVETraces : public CGradientVETraces
{
protected:
	CFeatureVFunction *featureVFunction;
	CStateProperties *featureProperties;


public:
	CFeatureVETraces(CFeatureVFunction *gradientVFunction);
	CFeatureVETraces(CFeatureVFunction *gradientVFunction, CStateProperties *featureProperties);


	/// Adds the state to the etrace List
	virtual void addETrace(CStateCollection *State, double factor = 1.0);
};


#endif

