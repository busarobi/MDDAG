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

#ifndef C_DISCRETETHEORETICALMODEL_H
#define C_DISCRETETHEORETICALMODEL_H

#include "clearndataobject.h"
#include "caction.h"
#include "cbaseobjects.h"
#include "cagentlistener.h"
#include "cutility.h"

#include <map>
#include <list>

#define TRANSITION 1
#define SEMIMDPTRANSITION 2

class CFeatureQFunction;
class CAbstractStateDiscretizer;
class CFeatureCalculator;
class CTransitionFunction;
class CFeatureList;

class CStateCollection;
class CState;
//class CMyArray2D<CStateActionTransitions *>;

/// Transition for the Markov Case
/**	
CTransition represents a transition for the Markov case. Stores the start-state, the end-state and the probability of the transition. 
The type of the transition is TRANSITION. With the type fiel you can determine wether its a CTransition or a CSemiMDPTransition.
*/

class CTransition
{
protected:
	int startState;
	int endState;
	double propability;
	
	int type;
public:
	CTransition(int startState, int endState, double prop);
	virtual ~CTransition() {};

	int getStartState();
	int getEndState();

	virtual double getPropability();
	virtual void setPropability(double prop);

	virtual void loadASCII(FILE *stream, int fixedState, bool forward);
	virtual void saveASCII(FILE *stream, bool forward);

	virtual bool isType(int Type);
};

/// Transition for the semiMDP case
/**
CSemiMDPTranstion stores a Transition for the Semi-Markov case. Additionally has an list of possible durations and the probabilities of the duration,
so the probability of the duration multiplied with the transition probability is the probability of coming from state A to state B in "duration" steps, executing the given action.
The type field of CSemiMDP transition is set to SEMIMDPTRANSITION.
When adding a duration with a specific factor, all other duration factors get multiplied by 1 - factor, and then the given duration's factor is
added or a nes duration is added to the list if the duration wasn't member. When setting a duration all other durations factors
are multiplied by 1 - (factor - factor_{old}).

*/
class CSemiMDPTransition : public CTransition
{
protected:
/// the list of the durations
	std::map<int, double> *durations;
public:
	CSemiMDPTransition(int startState, int endState, double prop);
	virtual ~CSemiMDPTransition();

	std::map<int, double> *getDurations();

/// adds the given factor the  duration's factor
/**
When adding a duration with a specific factor, all other duration factors get multiplied by 1 - factor, and then the given duration's factor is
added or a nes duration is added to the list if the duration wasn't member.
*/
	void addDuration(int duration, double factor);
/// adds the duration's factor to the given factor
	void setDuration(int duration, double factor);
	double getDurationFaktor(int duration);
/// returns the propability of transition with the specified duration
/** The propability is getPropability() * getDurationFaktor(duration)*/
	double getDurationPropability(int duration);

	virtual void loadASCII(FILE *stream, int fixedState, bool forward);
	virtual void saveASCII(FILE *stream, bool forward);
/** Returns the Faktor sum_N gamma^N*/
	double getSemiMDPFaktor(double gamma);
};


///Class for storing Transitions
/***The transitions are all stored in a CTransitionList object. The transition list stores whether it is a forward or a backward list. 
The Transitions are stored in a ordered list, the list is ordered by end-states for forward lists and by start-states for backward lists. 
It provides functions for adding a specific transition, getting the transition given a feature index and determining whether a feature index is member of the transition list. 
If the list is a forward list the search criteria for get and isMember is obviously the end-State otherwise the start-state of the transitions.
*/
class CTransitionList : public std::list<CTransition *>
{
protected:
/// flag if forward or backward List
	bool forwardList;

public:
	CTransitionList(bool forwardList);

/// Returns wether the feature is Member of the Transition
	/** I.e. if the list is a forward list it returns wether the featureIndex can be reached, fi the list is a backward list
	it returns wether the state "featuerIndex" can reach the assigned state from the List.
	*/
	bool isMember(int featureIndex);
	bool isForwardList();
/// Adds a transition
/*Adds a Transition to the sorted list in the right position*/
	void addTransition(CTransition *transition);
/// Returns the Transition with the specified feature as end (forward list) resp. start (backward list) state 
	CTransition *getTransition(int featureIndex);

	CTransitionList::iterator getTransitionIterator(int featureIndex);
/// Clears the List and deletes all CTransition objects
	void clearAndDelete();
};

///Class for storing the Backward and Forward Transitions for a given state action pair.
/**Saves the forward and the backward Transitions in 2 different Transition Lists.
*/
class CStateActionTransitions
{
protected:
	CTransitionList *forwardList;
	CTransitionList *backwardList;

public:
	CStateActionTransitions();
	~CStateActionTransitions();

	
	CTransitionList* getForwardTransitions();
	CTransitionList* getBackwardTransitions();
};


/// Interface for all model classes
/**The models are only designed for feature and discrete states (i.e. discretized states). The class defines the functions for getting the Probabilities of a specific state transition 
(so a state-action-state, resp. feature-action-feature tuple is given). In Addition you can retrieve the forward and the backward transitions for a specific state action pair.
<p>
The interface provides 4 Functions, which the subclasses have to implement:
<ul>
<li> getPropability(int oldFeature, int action, int newFeature) has to return the propability P(s'|s,a) </li>
<li> getPropability(int oldFeature, int action, int duration, int newFeature) has to return the propability for the semi MDP case, i.e. P(s',N|s,a) </li>
<li> CTransitionList* getForwardTransitions(int action, int state) has to return a list of all Transitions containing the states which can be reach from the state executing the action. </li>
<li> CTransitionList* getBackwardTransitions(int action, int state) has to return a list of all Transitions containing the states which can reach the state given state executing the action. </li>
</ul>
*/

class CAbstractFeatureStochasticModel : public CActionObject
{
protected:
	unsigned int numFeatures;
	CStateModifier *discretizer;

	bool createdActions;
public:
/// To create the model you have to provide the Models actions and the number of different states
	CAbstractFeatureStochasticModel(CActionSet *actions, int numStates);
	CAbstractFeatureStochasticModel(CActionSet *actions, CStateModifier *discretizer);
	CAbstractFeatureStochasticModel(int numActions, int numFeatures);
	virtual ~CAbstractFeatureStochasticModel();

	

/// Calls the getPropability function with the action index as argument
	virtual double getPropability(int oldFeature, CAction *action, int newFeature);
/// Interface function
/** 
has to return the propability P(s'|s,a)
*/
	virtual double getPropability(int oldFeature, int action, int newFeature) = 0;
/// Interface function
	/**has to return the propability for the semi MDP case, i.e. P(s',N|s,a)
*/
	virtual double getPropability(int oldFeature, int action, int duration, int newFeature) = 0;

///Calculates the propabilities for a list of features
	virtual double getPropability(CFeatureList *oldList, CAction *action, CFeatureList *newList);

	virtual double getPropability(CStateCollection *oldState, CAction *action, CStateCollection *newState);
	virtual double getPropability(CState *oldState, CAction *action, CState *newState);

/// Interface Function
/**
has to return a list of all Transitions containing the states which can be reach from the state executing the action.
*/
	virtual CTransitionList* getForwardTransitions(int action, int state) = 0;
	virtual CTransitionList* getForwardTransitions(CAction *action, CState *state);
	virtual CTransitionList* getForwardTransitions(CAction *action, CStateCollection *state);
/// Interface Function
/**
has to return a list of all Transitions containing the states which can reach the state given state executing the action.
*/

	virtual CTransitionList* getBackwardTransitions(int action, int state) = 0;
	virtual unsigned int getNumFeatures();
};



/// Class for loading and storing a fixed Model
/**The class CFeatureStochasticModel implements all functions from the interface CAbstractFeatureStochasticModel, therefore it maintains a CStateActionTransitions for every state-action pair.
For storing the CStateActionTransitions object a 2 dimensional array is used.
The class provides additional functions for setting the probability of a transition.
If the transition doesn't exist, a new transition object is created with the specified probability, otherwise the probability is just set. 
For the semi-MDP case, the duration of the transition can be specified as well. The given probability is then added to the existing transition probability and the duration factors
are all adopted so that there sum is again one. The model can only be used for loading, saving and directly setting the probabilities, its not able to learn anything from a learning trial. 
*/

class CFeatureStochasticModel : public CAbstractFeatureStochasticModel
{
protected:
/// The array of state-action Transitions	
	CMyArray2D<CStateActionTransitions *> *stateTransitions;
/// Load the model from file, can only be used at the constructor of the class.
	void loadASCII(FILE *stream);

/// Returns a new Transition object for the specified action
/** If the action is a multistep action, a CSemiMDPTransition is returned, otherwise a CTransition object. The transition
gets initialised with the given values.
*/
	CTransition *getNewTransition(int  startState, int endState, CAction *action, double propability);

	
public:
/// Loads the model from file.
	CFeatureStochasticModel(CActionSet *actions, int numFeatures, FILE *file);
/// Creates a new model
	CFeatureStochasticModel(CActionSet *actions, int numFeatures);
	CFeatureStochasticModel(int numActions, int numFeatures);
	virtual ~CFeatureStochasticModel();

/// returns the Propability of the transition
	/** Looks in the forward transitions of <oldFeature, action> wether a Transition to newFeature exists, if not 0 is returned, odtherwise
	the propybility of the transition.
	*/
	virtual double getPropability(int oldFeature, int action, int newFeature);
// returns the Propability of the transition
/** Looks in the forward transitions of <oldFeature, action> wether a Transition to newFeature exists with the specified duration, if not 0 is returned, odtherwise
the propybility of the transition.
*/
	virtual double getPropability(int oldFeature, int action, int duration, int newFeature);
	void setPropability(double propability, int oldFeature, int action, int newFeature);
	void setPropability(double propability, int oldFeature, int action, int duration, int newFeature);

/// Just returns thet forward trnasition list for the given state-action Pair.
	virtual CTransitionList* getForwardTransitions(int action, int state);
/// Just returns thet forward trnasition list for the given state-action Pair.
	virtual CTransitionList* getBackwardTransitions(int action, int state);

	virtual void saveASCII(FILE *stream);
};

class CStochasticModelAction : public CPrimitiveAction
{
protected:
	CAbstractFeatureStochasticModel *model;
public:
	CStochasticModelAction(CAbstractFeatureStochasticModel *model);
	virtual ~CStochasticModelAction(){};

	virtual bool isAvailable(CStateCollection *state);

};


/*
class CFeatureStateVisitCounter : public CLearnDataObject, public CStateObject, public CSemiMDPListener
{
protected:

};

class CFeatureStateActionVisitCounter : public CLearnDataObject, public CStateObject, public CSemiMDPListener
{
protected:

};*/

/// Base class for all estimated models.
/**
Estimated Models estimate the propability of the state transition by counting the number of Transitions from  a specific state action pair to a specific state and the number of visits from of the specific state-action pair. So
the estimated model is build on the fly, during learning. 
This is done by the class CAbstractFeatureStochasticEstimatedModel. The class is subclass of CFeatureStochasticModel so it stores the transition probabilities in the Transition list. In addition it has an double array which stores
the visits of the state action pair (double is needed because feature visits can be double valued). The Transition-visits are not stored explicitly but can be recovered by multiplying the 
probability with the visits of the state action pair. 
\par
The class CAbstractFeatureStochasticEstimatedModel provides the function doUpdateStep for updating the transitions and the visit table when a specific feature is visited (with an given factor). 
The function first calculates the visits of the Transitions (multiplying state-action visits with transition propability), updates the visits of the state-action pair (the factor of the feature is added), and then recalculates the new probabilities of the transitions (by dividing the transition visits through the
state action visits). Before this is done the feature factor is added to the specified transition's visits or a new Transition object is created if the transition hasn’t existed by now.
\par
The class also has the possibility to forget transitions from the past, so the propabilities can adapt to changing models more quickly. This is done
by the timeFaktor. Each time an update occurs, the state-actoin visits are multiplied by the timeFaktor before updating. By default the time factor is 1.0, so nothing is forgotten.
<p>
There are additional functions for retrieving the transition and the state action and the state visits. 
<p>
The subclasses of CAbstractFeatureStochasticEstimatedModel only have to implement the function nextStep(...) from the CSemiMDPListener interface. Indermediate steps don't need a special treatment, and 
are updated like normal step.
*/

class CAbstractFeatureStochasticEstimatedModel : public CFeatureStochasticModel, public CSemiMDPListener, public CStateObject, public CLearnDataObject
{
protected:

	CFeatureQFunction *stateActionVisits;
/// Updates the propabilities of the transitions from oldFeature and the given actions.
/**
The function first calculates the visits of the Transitions (multiplying state-action visits with transition propability), updates the visits of the state-action pair (the factor of the feature is added), and then recalculates the new probabilities of the transitions (by dividing the transition visits through the
state action visits). Before this is done the feature factor is added to the specified transition's visits or a new Transition object is created if the transition hasn’t existed by now. For the SemiMDP case the duration is added to the transition after the updates.
*/

	virtual void updateStep(int oldFeature, CAction *action, int newFeature, double Faktor);
public:
///Creates an new estimated model
	CAbstractFeatureStochasticEstimatedModel(CStateProperties *properties, CFeatureQFunction *stateActionVisits, CActionSet *actions, int numFeatures);
///Loads an estimated model from a file
	CAbstractFeatureStochasticEstimatedModel(CStateProperties *properties, CFeatureQFunction *stateActionVisits, CActionSet *actions, int numFeatures, FILE *file);
	
	virtual ~CAbstractFeatureStochasticEstimatedModel();

	/// the nextStep method, must be implemented by the subclasses
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState) = 0;
	/// intermediate Steps can be treated as normal steps in the model based case
	virtual void intermediateStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);

	virtual void saveData(FILE *stream);
	virtual void loadData(FILE *stream);

	virtual void resetData();

/// Returns the Transition visits of the specified Transition
/** The transition visits show how often a specific transition has been occured and they are calcualted by multiplying the propability of the transition with the visits of the state action pair.
*/
	double getTransitionsVisits(int oldFeature, CAction *action, int newFeature);
/// Returns the State Action Visits
/** Returns how often the given action was choosen in the given state. The State Action visits are stored in saVisits.
*/
	double getStateActionVisits(int Feature, int action);
/// Returns how often the agent visited the given state
/** This is calculated by summing up the state action visits.
*/
	double getStateVisits(int Feature);

};

/// Estimated Model for Discrete States
/**
Implements the fuction nextStep for updating.  CDiscreteStochasticEstimatedModel updates the transitions of the specified discrete state number with visit factor 1.0. 
To retrieve the state from the statecollection the in the constructor given discretizer is used.
@see CAbstractFeatureStochasticEstimatedModel
*/
class CDiscreteStochasticEstimatedModel : public CAbstractFeatureStochasticEstimatedModel
{
protected:
	CAbstractStateDiscretizer *discretizer;

public:
	CDiscreteStochasticEstimatedModel(CAbstractStateDiscretizer *discState, CFeatureQFunction *stateActionVisits, CActionSet *actions);
	virtual ~CDiscreteStochasticEstimatedModel() {};

/// Updates the discrete state transition
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);

	int getStateActionVisits(int Feature, int action);
	int getStateVisits(int Feature);
};

/// Estimated Model for feature States
/**Implements the fuction nextStep for updating. CFeatureStochasticEstimatedModel updates all transitions for all combinations of start and end-states.
The visit factor is calculated by multiplying the 2 feature factors of the specific start and end-features.
To retrieve the state from the state collection the in the constructor given feature calculator is used.

*/

class CFeatureStochasticEstimatedModel : public CAbstractFeatureStochasticEstimatedModel
{
protected:
	CFeatureCalculator *featCalc;

public:
	CFeatureStochasticEstimatedModel(CFeatureCalculator *properties, CFeatureQFunction *stateActionVisits, CActionSet *actions);
	virtual ~CFeatureStochasticEstimatedModel() {};
///Updates the Transitions for feature states
	/**Updates all transitions for all combinations of start and end-states.
	The visit factor used by doUpdateStep is calculated by multiplying the 2 feature factors of the specific start and end-features.
	To retrieve the state from the state collection the in the constructor given feature calculator is used.

	*/
	virtual void nextStep(CStateCollection *oldState, CAction *action, CStateCollection *nextState);
};



#endif

