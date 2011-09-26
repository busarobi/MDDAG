#include "cbaseobjects.h"
 
#include "cstatecollection.h"
#include "cstateproperties.h"
#include "cstatemodifier.h"
#include "caction.h"


CActionObject::CActionObject(CActionSet *actions, bool createNew)
{
	this->ownActionSet = createNew;
	if (createNew)
	{
		this->actions = new CActionSet();
		this->actions->add(actions);
	}
	else
	{
		this->actions = actions;
	}
}

CActionObject::~CActionObject()
{
	if (ownActionSet)
	{
		delete actions;
	}
}
	
CActionSet *CActionObject::getActions()
{
	return actions;
}

unsigned int CActionObject::getNumActions()
{
	return actions->size();
}

CStateObject::CStateObject(CStateProperties *properties)
{
	this->properties = properties;
}
	
CStateProperties *CStateObject::getStateProperties()
{
	return properties;
}

bool CStateObject::equalsModelProperties(CStateObject *object)
{
	return properties->equals(object->getStateProperties());
}

unsigned int CStateObject::getNumContinuousStates()
{
	return properties->getNumContinuousStates();
}

unsigned int CStateObject::getNumDiscreteStates()
{
	return properties->getNumDiscreteStates();
}

CStateModifiersObject::CStateModifiersObject(CStateProperties *modelState) : CStateObject(modelState)
{
	modifiers = new std::list<CStateModifier *>();
}

CStateModifiersObject::CStateModifiersObject(CStateProperties *modelState, std::list<CStateModifier *> *l_modifiers) : CStateObject(modelState)
{
	modifiers = new std::list<CStateModifier *>();
	addStateModifiers(l_modifiers);
}

CStateModifiersObject::~CStateModifiersObject()
{
	delete modifiers;
}

void CStateModifiersObject::addStateModifier(CStateModifier *modifier)
{
	modifiers->push_back(modifier);
}

void CStateModifiersObject::addStateModifiers(std::list<CStateModifier *> *modifiers)
{
	std::list<CStateModifier *>::iterator it = modifiers->begin();

	for (; it != modifiers->end(); it ++)
	{
		addStateModifier((*it));
	}	
}

void CStateModifiersObject::removeStateModifier(CStateModifier *modifier)
{
	modifiers->remove(modifier);
}

std::list<CStateModifier *> *CStateModifiersObject::getStateModifiers()
{
	return modifiers;
}

