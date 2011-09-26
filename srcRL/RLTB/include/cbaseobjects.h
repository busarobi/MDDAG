#ifndef C_BASEOBJECTS__H
#define C_BASEOBJECTS__H

class CStateModifier;
class CStateProperties;
class CStateCollection;
class CState;
class CAction;
class CActionSet;
class CActionData;
class CActionDataSet;

#include <list>
/// Base Class for all Classes which have to maintain an action set.
/** It just has a pointer to the actionset an some functions to get information
about the actionset */
class CActionObject  
{
	protected:
		CActionSet *actions;
		bool ownActionSet;
	public:
		CActionObject(CActionSet *actions, bool createNew = false);
		virtual ~CActionObject();

		CActionSet *getActions();
		unsigned int getNumActions();

};


/// Base class for all classes which have to manage with states.
/** The class just saves a state properties object pointer which can be used to retrieve a state from
a state collection.*/
class CStateObject
{
	protected:
// The properties of the wanted statea
		CStateProperties *properties;
	public:
		CStateObject(CStateProperties *properties);
	
		CStateProperties *getStateProperties();

/// Compares the two properties of the objects.
		bool equalsModelProperties(CStateObject *object);

/// Just returns the number of continuous states from the properties object.
		unsigned int getNumContinuousStates();
/// Just returns the number of discrete states from the properties object.
		unsigned int getNumDiscreteStates();
};

///Base Class for all Classes that have to maintain a list of modifiers

class CStateModifiersObject : public CStateObject
{
	protected:
		std::list<CStateModifier *> *modifiers;
	public:
		CStateModifiersObject(CStateProperties *modelState);
		CStateModifiersObject(CStateProperties *modelState, std::list<CStateModifier *> *modifiers);
	
		virtual ~CStateModifiersObject();


		virtual void addStateModifier(CStateModifier *modifier);
		virtual void addStateModifiers(std::list<CStateModifier *> *modifiers);

		virtual void removeStateModifier(CStateModifier *modfier);

		std::list<CStateModifier *> *getStateModifiers();

};


#endif

