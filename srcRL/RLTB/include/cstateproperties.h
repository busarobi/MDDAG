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

#ifndef C_MODELPROPERTIES_H
#define C_MODELPROPERTIES_H


#define FEATURESTATE 1
#define DISCRETESTATE 2
#define STATEDERIVATIONX 4
#define FEATURESTATEDERIVATIONX 5

/// Class defining the Properties of a State
/** The class contains all the Properties a single State can have. Each CState (actually even each CStateObject) has a pointer to his state properties.
The class contains the number of discrete and continuous states-variables, the discrete state size for each discrete state variable and the minimum and maximum 
values for the continuous states.
<p>
The member type describes the type of the state. There are three different types by now:
<ul>
	<li> 0 : "Normal" Modelstate: Can have an arbitrary number of discrete and continuous states, can't be used for the most qFunctions. </li>
	<li> DISCRETESTATE: Can only have one discrete state. These states are usually calculated by a CAbstractStateDiscretizer, not by the model.</li>
	<li> FEATURESTATE: Has to have the same number of discrete and continuous states. The i th continuous state coresponds to the i th discrete state, the discrete states determine
		the feature index and the continuous states the factor of that feature. All discrete state sizes have to be the same, all factors have to sum up to one, their minimum value is 0.0 and their max value is 1.0.</li>
</ul>
<p>
The state properties object are also very important for the CStateCollection class. Here the state properties object pointer
serves as Id to retrieve the state with the specific properties from the statecollection.
@see CStateObject
@see CState
@see CStateCollection
*/
class CStateProperties
{
protected:
/// number of continuous states
	unsigned int continuousStates;
/// number of discrete states
	unsigned int discreteStates;

/// type of the State
	int type;

/// an array containing the discrete state sizes
	unsigned int *discreteStateSize;

/// an array containing the minimum values of the continuous states
	double *minValues;
/// an array containing the maximum values of the continuous states
	double *maxValues;

	bool *isPeriodic;

	bool bInit;

	CStateProperties();

	virtual void initProperties(unsigned int continuousStates, unsigned int discreteStates,int type = 0);
public:

/// Creates a properties object with continuousStates continuous states and discreteStates discrete states.
/** The discrete state sizes, minimum and maximum values cant't be given to the constructor, these values have to be set
explicitly.*/ 
	CStateProperties(unsigned int continuousStates, unsigned int discreteStates,int type = 0);
/// Creates a properties object with the same properties as the given properties object.
	CStateProperties(CStateProperties *properties);
	virtual ~CStateProperties();

	int getType();
	bool isType(int type);

	/// adds a specific type to the type field bitmap
	/**
	So The parameter should be a power of 2, because al bits in the "Type" parameter gets set with an OR mask
	to the internal type.
	*/
	void addType(int Type);

/// Sets the discrete state size of the dim th state
	void setDiscreteStateSize(unsigned int dim, unsigned int size);
/// Returns the discrete state size of the dim th state
	virtual unsigned int getDiscreteStateSize(unsigned int dim);

/// Returns the number of continuous state variables
	unsigned int getNumContinuousStates();
/// Returns the number of discrete state variables
	unsigned int getNumDiscreteStates();

/// returns the discrete state size of all discrete states together (i.e the product of all sizes)
	virtual unsigned int getDiscreteStateSize();

/// sets the min-value of the dim-th continuous state
	void setMinValue(unsigned int dim, double value);
/// returns the min-value of the dim-th continuous state
	double getMinValue(unsigned int dim);

/// sets the max-value of the dim-th continuous state
	void setMaxValue(unsigned int dim, double value);
/// returns the max-value of the dim-th continuous state
	double getMaxValue(unsigned int dim);

	void setPeriodicity(unsigned int index, bool isPeriodic);
	bool getPeriodicity(unsigned int index);

	double getMirroredStateValue(unsigned int index, double value);
//	double getSingleStateDifference(int index, double difference);

/// Compares two properties in all their attributes, even discrete state sizes min and max values.
	bool equals(CStateProperties *object);
};



#endif

