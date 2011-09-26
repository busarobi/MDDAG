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


#ifndef C_PARAMETERS__H
#define C_PARAMETERS__H

#include <time.h>
#include <stdio.h>

#include <map>
#include <list>
#include <string>
#include <iostream>
#include <utility>

#define LINEAR 1
#define SQUARE 2
#define LOG	3
#define FRACT 4
#define FRACTSQUARE 5
#define FRACTLOG 6
#define EXP 7

using namespace std;

class CAdaptiveParameterCalculator;


/// This class represents a parameterset
/**
In the RL toolbox many objects, like learning algorithms have different parameters like the learning rate or the 
discount factor. The toolbox supports a general parameter handling for nearly all parameters of all the different classes. All these parameters can be set by a unified interface, which is supported by the class CParameters.
Every parameter in the toolbox is represented by its parameter name as a string and its parameter value. 
The toolbox provides methods to:
- set a Parameter value: In order to set a paramter just call the method setParameter(string parameterName, double value), with the name of the parameter and the value as argument. If you specify a parameter that doesn't exist you will get a warning message on the standard output, 
- get a Parameter value: Retrieve the value of a parameter with getParameter(string paramName). If you specify a name that doesn't exist you will get an assertation, so be careful when typing the names (or copy them).
- add a Parameter: A parameter can be added to the parameterset by the function addParameter, where you define the parameters name and its default value. If the parameter has already be added to the parameterset nothing happens (the default value won't be set too). Before a paramter is added, his value can't be set or retrieved.
Parameters are usually double values, but there are cases where integer or bool values are needed. If a parameter represents a integer value, it is rounded each time it is used. For boolean values, the >= 0.5 operator is used on the parameter value ( so 0.0 is false and 1.0 is true).
If you want to know more about the parameters of an object please consult the class reference of the specific object. The parameters set of an object can also be printed to standard output by calling the method saveParameters(stdout), which is a good way to see which parameters an specific object has.
The name and the value are stored in the parameters map of the CParameter object.
The class also provides methods for saving and loading parameter sets. You have the possibility to read/write the parameterset from/to a FILE stream, or a stream object. There is also the possibility to save the parameterset in xml style (without header).
For more details on parameter handling see the class reference of CParameterObject or the Manual.
*/

class CParameters
{
protected:
	std::map<string, double> *parameters;
	std::map<string, bool> *isAdaptive;
public:
	/// Create new parameter set
	CParameters();

	/// Create new parameter set and copy the given reference
	CParameters(CParameters &copy);

	virtual ~CParameters();

	bool containsParameters(CParameters *parameters);
		
	void loadParameters(FILE *stream);
	void saveParameters(FILE *stream);

	void loadParametersStream(istream *stream);
	void saveParametersStream(ostream *stream);

	/// save in XML-style
	/**
	 This is just pseudo-xml, so ne formatting rules or header is written.                                                                  
	*/
	void saveParametersXML(FILE *stream);
	/// load from XML-style stream
	void loadParametersXML(FILE *stream);

	/// Add the paramter with the given name and set value as default
	/**
	If the parameter is already in the parameter set, the default value is not set!
	*/
	virtual void addParameter(string name, double value);

	/// Add all paramters of the given parameter set to the current parameter set
	virtual void addParameters(CParameters *parameters);

	/// Remove parameter from set
	virtual void removeParameter(string name);

	/// returns the parameter value
	/**
	Attention: throws an assertation if the parameter is unknown!!!
	*/
	virtual double getParameter(string name);
	
	/// Set the parameter to the given value
	/**
	Prints a warning if the parameter is unknown
	*/
	virtual void setParameter(string name, double value);
	
	/// sets all parameters which are in both parameter set to the values of the given action set
	virtual void setParameters(CParameters *parameters);

	/// get parameter value from its index
	double getParameterFromIndex(unsigned int index);
	/// get parameter name from its index
	string getParameterName(unsigned int index);
	/// set parameter value with its index
	void setParameterWithIndex(unsigned int index, double value);

	/// get the index of the parameter
	int getParameterIndex(string name);

	int getNumParameters();

	void setAsAdaptiveParameter(string parameter, bool adaptive);
	bool isAdaptiveParameter(string parameter);
	
	/// returns true if all parameters are the same
	virtual bool operator == (CParameters &parameters);
	/// compares 2 parameter sets (needed for the parameter maps)
	bool operator < (CParameters &paramters);
};

/// Super class of all Objects mantaining parameters
/**
This class provides full parameter handling support for all its subclasses. Additionally to the functionality of the CParameters class, this class supports adding parameters of other CParameterObject's to the parameter set. The difference to normal to adding normal CParameter objects is, that if the parameter of the current object changes, all parameter objects which have been added by addParameters will get informed about the changed parameter. So if several objects have the same parameter you only have to set it once. Therefore the class contains a parameter object list to inform the other parameter objects
In the toolbox all parameter objects which are data elements of another parameter object get added to the parameter object itself, so all parameters of the data elements are parameters of the new class too. So for example the TD-Learner class contains the parameter "Lambda", even though this parameter initially belonged to the etrace object of the learner. If we change the parameter "Lambda" of the TD-Learner, it will also change the parameter "Lambda" for its etrace object. 
If 2 or more data elements have the same parameter, they can only have the same parameter value, because all of the parameter objects get informed about a parameter change. If this isn't desired, you can specify a parameter name prefix, when adding the parameter object to your new class. This Prefix is used to distinguish between the same parameter names of the parameter objects. Per default no prefix is used.
An additional functionality of parameter objects are adaptive paramter. For each parameter you can specify an adaptive parameter calculator, which calculates the parameter value each time it is retrieved. Now, each time the parameter's value is requested by "getParameter" the calculated value of the adaptive parameter calculator is returned instead of the constant double value of the parameter map. This is useful for example for adapting the learning rate or the exploration of a policy. The parameter's value can depend on any other value like the number of steps or episodes or even the current average reward. (see CAdaptiveParameterCalculator). Be aware that the adaptive parameter calculator is always set only for current object in the parameter object hierarchy. So if you set an adaptive parameter for the Parameter "Lambda" in a TD-Learner object, it won't affect the etraces, where the paremeter initially belong. So you have to set the adaptive parameter for the etrace object directly.
For performance reasons the parameter object subclasses have the possibility to not use the parameter set everytime they want to retrieve the parameters value, therefore they can store the parameters in double values, and each time a parameter value changes, they get informed by the function onParametersChanged. So this function has to be overwritten to update the double values if this is needed.
*/

class CParameterObject : public CParameters
{
protected:
	/// informs all parameter objects from the list and calls onParametersChanged
	void parametersChanged();
	

	typedef std::pair<CParameterObject *, string> paramPair;
	
//	std::map<string, CAdaptiveParameterCalculator *> *adaptiveParameters;

	std::list<paramPair> *parameterObjects;
public:
	CParameterObject();
	virtual ~CParameterObject();

	/// Interface for faster parameter handling (see description of the class)
	virtual void onParametersChanged() {};

	/// sets the parameter and calls parametersChanged
	virtual void setParameter(string name, double value);
	/// sets the parameters and calls parametersChanged
	virtual void setParameters(CParameters *parameters);

	/// Add all parameters of the given parameter object to the current object, also add the given parameter object to the parameter object list
	/**
	All parameters of the given object gets added with the prefix to the parameter set.
	*/
	virtual void addParameters(CParameterObject *parameters, string prefix = "");

	/// reset all adaptive parameter calculators, this is needed when you want to restart learning.
	/** 
	Calls resetCalculators for all adaptive Parameters in the map.
	*/
	//virtual void resetParameterCalculators();

	/// Returns the parameters value
	/**
	If there is the specified parameter is not adaptive, the function returns the constant parameter value (see CParameters). Otherwise the value is calculated by the adaptive parameter calculator.
	*/
	virtual double getParameter(string name);

	/// Add an adaptive parameter calculator for a given parameter
	/**
	 The adaptive Parameter calculator is added to the adaptiveParameters map, so getParameter can check wether an adaptive parameter calculator is defined for the specified calculator.
	 Be aware that the adaptive parameter calculator is always definied only for the current object in the parameter object hierarchy. So if you set an adaptive parameter for the Parameter "Lambda" in a TD-Learner object, it won't affect the etraces, where the paremeter initially belong. So you have to set the adaptive parameter for the etrace object directly. The calculators often have parameters themself (like the parameter scale or offset), these parameters are certainly added to the parameter object. When an adaptive parameter calculator is set to a parameter, the parameters name is used as prefix for the parameters of the adaptive parameter calculator.  
	 */
	//virtual void addAdaptiveParameter(string name, CAdaptiveParameterCalculator *paramCalc);
	/// Remove the adaptive parameter again
	/**
	The parameters constant value is used again.
	*/
	//virtual void removeAdaptiveParameter(string name);

	/// returns true if all parameters are the same
	virtual bool operator == (CParameters &parameters);
};

/// Interface for all adaptive Parameter Calculators
/** For each parameter you can specify an adaptive parameter calculator (APC), which calculates the parameter value each time it is retrieved. Now, each time the parameter's value is requested by "getParameter" the calculated value of the adaptive parameter calculator is returned instead of the constant double value of the parameter map. This is useful for example for adapting the learning rate or the exploration of a policy. The parameter's value can depend on any other value like the number of steps or episodes or even the current average reward.
Adaptive Parameter Calculators also have same parameters too, all parameters of the Adaptive Parameter Classes begin with the prefix "AP". When an adaptive parameter calculator is set to a parameter, the parameters name is used as prefix for the parameters of the adaptive parameter calculator. So the parameter "APFunctionKind" gets to the parameter "VLearningRateAPFunctionKind" if you specify a APC for the parameter "VLearningRate".
The interface CAdaptiveParameterCalculator already includes the parameter "APFunctionKind", the functionkind property is used to determine which function shall be used to transform the targetvalue in the parametervalue. The targetvalue can be the number of learning steps, number of episodes, the current value of a V-Function or the average reward. See the subclasses for more details. There are 6 different functionkinds implemented.
- Linear Function (LINEAR, 1) 
- Square Function (SQUARE, 2)
- Logarithm Function (LOG, 3)
- Fraction (FRACT, 4)
- Squared Fraction (FRACTSQUARE, 5)
- Logarithm Fraction (FRACTLOG, 6)
All these functions are used in a slightly different way for the 2 main subclasses, CAdaptiveParameterBoundedValuesCalculator and CAdaptiveParameterUnBoundedValuesCalculator. For more details see these classes.
Parameters of CAdaptiveParameterCalculator:
"APFunctionKind": Defining the function to transform target value into the parameter value.
@see CAdaptiveParameterUnBoundedValuesCalculator
@see CAdaptiveParameterBoundedValuesCalculator
*/
class CAdaptiveParameterCalculator : virtual public CParameterObject
{
protected:
	/// The targetvalue is stored here
//	double targetValue;

	/// The function kind is stored here
	/**
	 The parameter "APFunctionKind" isn't used for performance reasons, functionKind is updated each time the "APFunctionKind" parameter changes (in the function onParameterChanged())
	 */
	int functionKind;

	CParameters *targetObject;
	string targetParameter;
public:
	CAdaptiveParameterCalculator(CParameters *targetObject, string targetParameter, int functionKind);
	virtual ~CAdaptiveParameterCalculator();

	/// Interface for all adaptive Parameter Calculators
	virtual void setParameterValue(double value);

	/// Reset the targetValue 
	/**
	This function is used for resetting for example the steps or number of episodes when learning is restarted. (used for parameter evaluation)
	*/
	virtual void resetCalculator() = 0;
	/// Updates functionKind according to the parameter "APFunctionKind"
	virtual void onParametersChanged();

};

/// Super class for all classes which use bounded target values
/**
The subclasses of theses class use bounded target values. These are for example the average reward or the value of a V-Function. For Bounded Target values you can define a minimum and a maximum value of the target (Parameters: "APTargetMin", "APTargetMax"). For example if the reward is supposed to be between -1 and 0 you can define these values as minimum and maximum target values for the average reward adaptive parameter calculator. The intervall [targetmin, targetmax] of the target value gets normalized to the intervall [0,1]. This intervall can be scaled by the parameter "APTargetScale". After the normalization the function defined by the parameter functionKind gets applied. The 6 different functions are calucalated the following way:
- LINEAR: f(x) = x
- SQUARE: f(x) = x^2
- LOG: f(x) = log(x * targetScale + 1.0) / log(1.0 + targetScale);
- FRACT: f(x) = f(x) = (1.0 / (x * targetScale + 1.0) - 1.0 / (targetScale + 1.0)) * (1.0 + targetScale) /targetScale;
- FRACTSQUARE: f(x) = (1.0 / (x^2* targetScale^2 + 1.0) - 1.0 / (targetScale^2 + 1.0)) * (1.0 + targetScale^2) /targetScale^2;
- FRACTLOG : offset = 1.0 / (1.0 + log(1.0 + targetScale));
			 f(x) = (1.0 / (1.0 + log(x * targetScale + 1.0)) - offset) / (1 - offset);
All the functions are scaled so that there function values are again in the intervall [0,1]. So scaling the targetintervall is only useful if log or fract functions are used (so you can set the steepness of the slope of this functions).
The result can be inverted (1 - x) if the Parameter "APInvertTargetFunction" is true (1.0). This value is now scaled ("APParamScale") and an offset gets added ("APParamOffset"), so the resulting parameter value is calculated with the formular param = param_offset + param_scale * f(normalized_targetvalue), resp. param = param_offset + param_scale * (1 - f(normalized_targetvalue)). These gives you much degree of freedom to design your adaptive parameter calculator.
The values of the parameters "APInvertTargetFunction", "APParamScale", "APParamOffset", APTargetMin" and "APTargetMax" are again stored in own data element for performance reasons and updated by the function onParameterChanged.
See the subclasses for the different target values.
Parameters of CAdaptiveParameterBoundedValuesCalculator:
- "APFunctionKind": Defining the function to transform target value into the parameter value.
- "APInvertTargetFunction": Boolean value wether to invert target function or not
- "APParamScale": Scale of the parameter value
- "APParamOffset": Parameter Value offset
- "APTargetMin": Minimum value of the target
- "APTargetMax": Maximum value of the target
- "APTargetScale": Scale for the targetValue, so the targetValue is in the intervall [0, targetScale].
*/
class CAdaptiveParameterBoundedValuesCalculator : public CAdaptiveParameterCalculator
{
protected:
	double targetMin;
	double targetMax;

	double targetScale;

	double paramOffset;
	double paramScale;

	bool invertTarget;

public:
	CAdaptiveParameterBoundedValuesCalculator(CParameters *targetObject, string targetParameter, int functionKind, double paramOffset, double paramScale, double targetMin, double targetMax);
	virtual ~CAdaptiveParameterBoundedValuesCalculator();

	/// Sets the targetValue to the targetMin value
//	virtual void resetCalculator();
	/// Updates all data elements represents parameters
	virtual void onParametersChanged();

/// Returns the value of the parameter
/**
The value of the parameter is calculated the follwing way:
- param = param_offset + param_scale * f(normalized_targetvalue)
- param = param_offset + param_scale * (1 - f(normalized_targetvalue)) for inverted function values (APInvertTargetFunction)
For more details see class description.
*/
	virtual void setParameterValue(double value);

};

/// Super class for all classes which use unbounded target values
/**
The subclasses of theses class use unbounded target values. These are for example the number of steps or the number of learned episodes. For unbounded target values you can define an offset and a scale value for the target (Parameters: "APTargetMin", "APTargetMax"). The target value is then transformed the following way x = target_offset + target_scale * target. After the transformation the function defined by the parameter "APFunctionKind" gets applied. The 6 different functions are calucalated the following way:
- LINEAR: f(x) = x
- SQUARE: f(x) = x^2
- LOG: f(x) = log(x + 1.0)
- FRACT: f(x) = (1.0 / (x + 1.0))
- FRACTSQUARE: f(x) = (1.0 / (x^2 + 1.0)
- FRACTLOG : f(x) = 1.0 / (1.0 + log(x + 1.0));
The result can be now again scaled and an offset can be added ("APParamScale", "APParamOffset"), so the resulting parameter value is calculated with the formular param = param_offset + param_scale * f(transformed_targetvalue). These gives you much degree of freedom to design your adaptive parameter calculator.
The values of the parameters "APParamScale", "APParamOffset", APTargetMin" and "APTargetMax" are again stored in own data element for performance reasons and updated by the function onParameterChanged.
See the subclasses for the different target values.
Parameters of CAdaptiveParameterBoundedValuesCalculator:
- "APFunctionKind": Defining the function to transform target value into the parameter value.
- "APParamScale": Scale of the parameter value
- "APParamOffset": Parameter Value offset
- "APTargetScale": Scale value of the target
- "APTargetOffset": Offset value of the target
*/
class CAdaptiveParameterUnBoundedValuesCalculator : public CAdaptiveParameterCalculator
{
protected:
	double targetOffset;
	double targetScale;

	double paramOffset;
	double paramScale;

	double paramLimit;

public:
	CAdaptiveParameterUnBoundedValuesCalculator(CParameters *targetObject, string targetParameter, int functionKind, double param0, double paramScale, double targetOffset, double targetScale);
	virtual ~CAdaptiveParameterUnBoundedValuesCalculator();

	/// Updates all data elements which represents parameters
	virtual void onParametersChanged();

	/// Returns the value of the parameter
	/**
	The value of the parameter is calculated the follwing way:
	- param = param_offset + param_scale * f(target_offset + target_scale * target)
	For more details see class description.
	*/
	virtual void setParameterValue(double value);
};


#endif
