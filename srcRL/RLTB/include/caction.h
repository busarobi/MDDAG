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

#ifndef CACTION_H
#define CACTION_H

#include <stdio.h> 
#include <vector>
#include <list>
#include <map>

#include "cbaseobjects.h"

class CStateCollection;
class CStateProperties;

#define EXTENDEDACTION 1
#define MULTISTEPACTION 2
#define PRIMITIVEACTION 4
#define CONTINUOUSACTION 16
#define CONTINUOUSSTATICACTION 32

/// interface for saving changable data of an action
/** Since the "Id" of an action is the pointer itself, and this pointer is used in every action set, we can't set the changeable data of an action that easily, since it would change the action's data everywhere in the program. If this isn't wanted (as usual), we have to use action data objects. New action data objects can be retrieved from any action class (if there is no changeable data for an action the action will return a NULL pointer). This action data object is now local, and not global any more and can therefore be changed easily. If an action has changeable data, the action pointer and a coresponding action data object determine the action. This is used in many function-arguments. Normally, if the action-data object is missing (= NULL) a new action data object is retrieved from the action.
\par
At the momemt its  used for saving the data of MultiStepActions (duration, finished), and ContinousActions (saving the continuous action value)
all other classes return NULL when asking them for a CActionData Object. 
@see CAction
@see CMultiStepActionData
@see CActionList
*/
class CActionData
{
protected:
	bool bIsChangeAble;
public:
	CActionData();
	virtual ~CActionData() {};
	
	virtual void saveASCII(FILE *stream) = 0;
	virtual void loadASCII(FILE *stream) = 0;

	virtual void saveBIN(FILE *stream) = 0;
	virtual void loadBIN(FILE *stream) = 0;

	/// Set the changeable data according to the actiondata object
	virtual void setData(CActionData *actionData) = 0;

	bool isChangeAble();
	void setIsChangeAble(bool changeAble);
};


/// class for saving the duration and the finished flag from a MultiStepAction
/**
@see CActionData
*/
class CMultiStepActionData : public CActionData
{
public:
	CMultiStepActionData();
	virtual ~CMultiStepActionData() {};

	int duration;
	bool finished;

	virtual void saveASCII(FILE *stream);
	virtual void loadASCII(FILE *stream);

	virtual void saveBIN(FILE *stream);
	virtual void loadBIN(FILE *stream);

	/// Set the changeable data according to the actiondata object
	virtual void setData(CActionData *actionData);

};



class CContinuousActionProperties;
class CActionDataSet;


///Class Representing an action the agent or any other SMDP can choose from. 
/** 
The class represents an interface for all kind of actions. The interface consists of an action type, methods for handling the action data object of the action and the isAvailable Method.
The action itself is identified with the pointer of the action abject together with an specific actionSet. So if you want to know the index of an action, you first must have an action set and then call its getIndex(CAction *) method. 
The class is just an interface for all other actions. It maintains type field to determine which type the action is. There are
following types:
<ul>
<li> EXTENDEDACTION </li>
<li> MULTISTEPACTION </li>
<li> PRIMITIVEACTION </li>
<li> CONTINUOUSACTION </li>
<li> CONTINUOUSSTATICACTION </li>
</ul>

Each type belongs to a specific class. Don't add a type if the class isn't subclass of the class belonging to the type! This will lead to type cast errors. If the type is added the class has to be cast-able to the class representing the type.
\par 
All the direct subclasses of CAction have CAction as a virtual base class, so you can combine the attributes of the different actions.
For example you can create an PrimitiveAction which can last for several steps (deviated from CPrimitiveAction and CMultiStepAction). Be carefull, due to the virtual base class you ALWAYS have to do a dynamic_cast instead of a normal type cast if you want to cast any CAction class object.
\par
It also has the member isAvailable(CStateCollection *), which is per default always true. With this function you can exclude
specific actions in some states, when they are not avialable for the agent. Befor choosing an action, each Controller first determines
 the list of available actions from its action set to choose a certain action. 
 Since the Method isAvailable gets an state collection you don't have to use the model state, you also have acces to other modified states (for example a discrete state). 
The index of action can't be identified by the action itself, you need always an actionset to get the index of the action.
*/
class CAction
{
protected:
/**
The type field is an integer, but it serves as a bitmask. So if you want to add an specific type to the Action 
you have to add it with an or mask. If you want to add a new general typ of Actions you have to create a new Typenumber. 
Since the bitmask only 2^n numbers are allowed.
*/
	int type;
	CActionData *actionData;

	CAction(CActionData *actionData);

public:
    CAction();
	virtual ~CAction();
	
    int getType();
	bool isType(int type);

/// adds a specific type to the type field bitmap
/**
So The parameter should be a power of 2, because al bits in the "Type" parameter gets set with an OR mask
to the internal type.
*/
	void addType(int Type);

	
	/// Returns an new CActionData object of the specific sub-class.
	/**
	The CActionData Object is created with new and must be deleted by the programmer!
	*/
	virtual CActionData *getNewActionData();

	
	/// Set the changeable data according to the actiondata object
	virtual void loadActionData(CActionData *actionData);
	/// Returns an actiondata object initialised with the actions values
	/**
	The CActionData Object is created with new and must be deleted by the programmer!
	*/
	virtual CActionData * getActionData();

	

	virtual bool isAvailable(CStateCollection *) {return true ;};//return isAvailable(state->getState());};
	//virtual bool isAvailable(CState *state) {return true;};

	///returns the duration of the action, per default 1
	virtual int getDuration() {return 1;};

	///Determines wether the 2 actions represent the same action
	virtual bool equals(CAction *action);

	/// Compares the action and the actionData obejcts.
	virtual bool isSameAction(CAction *action, CActionData *data);


};

/// A list of Actions representing the actual hierarchical Stack
/** The actual hierarchical stack contains alls ExtendedActions which are aktiv at the moment and the 
PrimitiveAction which is returned by the last extended action. 
*/
class CHierarchicalStack: public std::list<CAction *> 
{
protected:
	
public:
	CHierarchicalStack();
	virtual ~CHierarchicalStack();
	
	void clearAndDelete();
};


// An action which can long for several steps.
/** An action with a duration different than one has to be treated with SemiMarkov-Learning rules in all
the learning algorithms, since otherwise the result can be far from optimal. Many Learning-Algorithm in the this package support
Semi-Markov Learning updates.
This action class maintains its own action data object, a CMultiStepActionData object. This data object stores the duration it has needed by now and the finished flag. The duration and the finished flag get normally set by an HierarchicalController, but it is also possible to set the duration for example in the environment model, if you have a primitive action which takes longer than other primitive actions. In that case your model-specific action has to be derivated from CPrimitiveAction and CMultiStepAction.
\par
The class also contains the method isFinished(CStateCollection *state). This method is used (normally by a hierarchical controller) to determine
wether the action has finished or not. The controller sets the finished flag according to isFinished, so other Listeners only have to
look at this flag. This method must be implemented by all (non-abstract) sub-classes.
*/

class CMultiStepAction :  public CAction
{
protected:
	CMultiStepActionData *multiStepData;

	CMultiStepAction(CMultiStepActionData *multiStepData);
public:
	CMultiStepAction();
	virtual ~CMultiStepAction(){};

	/**
	This method is normally used by a hierarchical controller to determine
	wether the action has finished or not. The finsished method may depend only on the current state transition, so you get the old state and the new state as parameters. The controller sets the finished flag according to isFinished, so other Listeners only have to
	look at this flag. This method must be implemented by all (non-abstract) sub-classes.
	*/
	virtual bool isFinished(CStateCollection *oldState, CStateCollection *newState) = 0;

	virtual CActionData *getNewActionData();

	virtual CMultiStepActionData *getMultiStepActionData() {return multiStepData;};

	/// returns the duration of the action (member: duration)
	virtual int getDuration() {return multiStepData->duration;};
};

// Represents a primitive Action
/**
The only kind of actions which can be added to the Agent and passed to the EnvironmentModel as action to execute.
For a specific learning problem you have to derivate your ModelActions from this class and add some specific attributes
to the action (for example force...).
The type PRIMITIVEACTION is added to the type field of the action.
@see CPrimitiveActionStateChange
*/

class CPrimitiveAction : public CAction
{
protected:
	CPrimitiveAction(CMultiStepActionData *actionData);
public:
	CPrimitiveAction();
	virtual ~CPrimitiveAction();
	
	virtual int getDuration() {return 1;};
};

/// This abstract class represents extended actions like behaviors or hierarchical SMDPs.
/** The CExtendedAction class can represent behaviors and other actions which are composed of
other primitive, or even extended actions. Since an extended action usually consists of a composition of several other, "more primitive" actions, the extended action 
has also an duration, so its derivated from CMultiStepAction. Like an agent controller you can retrieve an other action from the
extended action to execute (with getNextHierarchyLevel(...)). The action returned by this method can be a primitive action or aswell an extended action with lower hierarchy. 
Gathering the next action until a primitiv action occurs is done by the class CHierarchicalController, 
meanwhile the Hierarchical Stack is also created. The Hierachic Controller also sets the nextHierarchyLevel Pointer according to getNextHierarchyLevel(...).
<p>
When using extended actions you have the possibility that all intermediate steps which occured during the execution of the
extended action get send to the Listeners by the class CSemiMarkovDecisionProcess. For sending the intermediate steps to a agent listener the function "intermediateStep" is used instead of "nextStep", the different function is needed because intermediate steps has to be treated differently in some cases (ETraces). To get the intermediate steps of a executed behavior you take all states 
occured during the execution of the action. To create a S-A-S tuple you take one state of that list of states for the first state in the tuple (soo its the "current" state), set the duration of the extended action correctly and 
for the second state of the tuple you take the state in which the action has finished. This is only usefull for behaviors 
which finishing condition only depends on the actual state. With this method your are able to provide more training examples.
If you don't wan
*/
class CExtendedAction : public CMultiStepAction
{
protected:
	CExtendedAction(CMultiStepActionData *actionData);


public:
	/// Pointer to the action executed by the extended Action.
	CAction *nextHierarchyLevel;

	CExtendedAction();
	virtual ~CExtendedAction(){};

/// Virtual function for determining the next action in the next hierarchie level.
	virtual CAction* getNextHierarchyLevel(CStateCollection *state, CActionDataSet *actionDataSet = NULL) = 0;
/// Constructs a hierarchical ActionStack, with the extended action itself as root.
	void getHierarchicalStack(CHierarchicalStack *actionStack);
/// Flag for sending the intermediate Steps of this action.
	bool sendIntermediateSteps;
};




/// class maintaining all the actions available for a certain object (normally a CActionObject like a controller) 
/** This class maintains a list of all actions which should be useable for another object.
It only saves the pointers of the action Objects, so you can't save for example the duration of an action, if
you change the duration later (which is normally the case), for that case use CActionList. The pointer of the action is also used as a kind of "Id". According 
to the pointer the function getIndex(...) returns the index of the action, so the other objects can determine which 
action was chosen.
/par 
CActionSet also provides a function for gettíng all available actions in the current State from the action set.
@see CActionList
*/
class CActionSet : public std::list<CAction *>
{

public:
	CActionSet();
	virtual ~CActionSet();

/// returns the index of the action.
	int getIndex(CAction *action);

/// returns the index of the index th action.
	CAction *get(unsigned int index);

/// add an action to the set.
	void add(CAction *action);

/// add all actions from an action set to the action set
	void add(CActionSet *actions);

/// get all available actions in the current State from the action set.
	void getAvailableActions(CActionSet *availableActions, CStateCollection *state);

//	returns wether the given action is member of the actionset
	bool isMember(CAction *action);
};


/// class for mantaining the action data objects of all actions of an action set
/** This class is needed mainly for controllers, since they are not allowed to change the data of the action itself, they maintain a local action dataset, and modify the data of the data set. 
*/
class CActionDataSet //: public CActionObject
{
protected:
	std::map<CAction *, CActionData *> *actionDatas;
public:
	CActionDataSet(CActionSet *actions);
	CActionDataSet();
	virtual ~CActionDataSet();

	CActionData *getActionData(CAction *action);
	void setActionData(CAction *action, CActionData *data);

	void addActionData(CAction *action);
	void removeActionData(CAction *action);
};

/// class for logging a sequence of actions
/** This class saves the ActionData Objects and the index of the actions in separate lists. 
If you add an action, the index is stored in the index list (so the action has to be member of the 
actionlist's actionset. If the specific action returns a valid CActionData Object, this is also stored in the actionDatas map.
/par
If you want to get an action with a specified number in the sequence, the action with the index specified by 
actionIndices[num] is returned, but before if there is an action data, it is set. 
*/
class CActionList : public CActionObject
{
protected:
	// vector for the action indices
	std::vector<int> *actionIndices;
	/** map for the actionDatas, the mapIndex is the actionNumber in the sequence, because not all action have an action data*/
	std::map<int, CActionData *> *actionDatas;
public:
	CActionList(CActionSet *actions);
	virtual ~CActionList();


	void addAction(CAction *action);
	CAction *getAction(unsigned int num, CActionDataSet *l_data);

	virtual void loadBIN(FILE *stream);
	virtual void saveBIN(FILE *stream);

	virtual void loadASCII(FILE *stream);
	virtual void saveASCII(FILE *stream);
	
	/// Returns the number of actions in the action list.
	unsigned int getSize();
	unsigned int getNumActions();
	

	void clear();
};



#endif



