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

#ifndef C_STATECOLLECTION_H
#define C_STATECOLLECTION_H


#include "cbaseobjects.h"
#include "cstate.h"

#include <list>
#include <map>

/// Implementation of the CStateCollection interface
/**
A state collection contains the "basic" state, usually the model state,
and a list of modified states with their state modifiers. Any component which has access to a statecollection (usually a listener), can retrieve
a state from the state collection as long as he mantains the pointer to the properties of the desired state, which serves as the id of the state 
in the state collection. When you add a state modifier, a state with the properties of the modifier is created ans stored in a map (modifiedStates).
Each time you change the model state you have to call the function newModelState(), the collection sets then all modified states as depricated. The modified state
gets recalculated when it is requested for the first time and the deprecated flag of the state gets cleared.
<p>
You can also set the modified states directly without calculation, this is useful if the modified state is already 
available by a logger.
*/

class CStateCollectionImpl : public CStateModifiersObject, public CStateCollection
{
protected:
	/// basic model state of the collection
	CState *modelState;
	/// stores all state modifiers with their states
	std::map<CStateModifier *, CState *> *modifiedStates;
	/// depricated flags
	std::map<CStateModifier *, bool> *stateCalculated;

public:
/// create state collection with the given state properties as basic state
	CStateCollectionImpl(CStateProperties *modelProperties);
/// copy state collection
	CStateCollectionImpl(CStateCollectionImpl *stateCollection);
/// create state collection with the given state properties as basic state and the given modifiers already added.
	CStateCollectionImpl(CStateProperties *properties, std::list<CStateModifier *> *modifiers);
	virtual ~CStateCollectionImpl();

/// calculate all modified states which are depricated
	void calculateModifiedStates();
/// set the state in the state collection with the same properties like the given state
	void setState(CState *state);
	
	void setStateCollection(CStateCollection *stateCollection);

/// get state with the given properties
	virtual CState *getState(CStateProperties *properties);
/// get basic state
	virtual CState *getState();
/// add state modifier
	virtual void addStateModifier(CStateModifier *modifier);
/// remove state modifier
	virtual void removeStateModifier(CStateModifier *modfier);

/// returns wether the state has already been calculated
	virtual bool isStateCalculated(CStateModifier *);

	/// returns wether the state has already been calculated
	virtual void setIsStateCalculated(CStateModifier *modifier, bool isCalculated);

/// returns the state Object for extern setting
//	virtual CState *returnStateForExternSetting(CStateProperties *);

/// returns wether the modifier states or the basic state have the specified properties
	virtual bool isMember(CStateProperties *stateModifier);

/// marks all modified states as depricated, forces them to recalculate
	void newModelState();
	
	virtual void setResetState(bool reset);
};

/// Class for storing a sequence of state collections
/** This class is able to store a sequence of state collections, all state collections added to the list are supposed to have
the same state modifiers. For storing the collection it uses a list of CStateList objects, for each state in the collection there is
a state list. When a state collection is added, the collection is split into its states and the states are added to the state lists. 
<p>
When retrieving a state collection from the list, only the states which are member of the given state collection get set in the collection,
no new state modifiers get added. 
@see CStateList
*/

class CStateCollectionList : public CStateModifiersObject
{
protected:
/// the list of state lists
	std::list<CStateList *> *stateLists;
	
     
public:
/// create a state collection list with "model" as the basic state for the collections
	CStateCollectionList(CStateProperties *model);
/// create a state collection list with "model" as the basic state for the collections and the modifiers already added
	CStateCollectionList(CStateProperties *model, std::list<CStateModifier *> *modifier);

	virtual ~CStateCollectionList();
/// clears alls state lists
	void clearStateLists();

/// add state collection to the list. 
/**the collection is split into its states and the states are added to the statelists. */
	void addStateCollection(CStateCollection *stateCollection);
/// retrieve collection from the list. 
/**When retrieving a state collection from the list, only the states which are member of the given state collection get set in the collection,
no new state modifiers get added*/
	void getStateCollection(int index, CStateCollectionImpl *stateCollection);

	void removeLastStateCollection();
	
	CStateList *getStateList(CStateProperties *properties);

/// get a State directly from the collection list, without the need of an state collection.
	void getState(int index, CState *state);

/// add a state modifier to the collection list.
/** An own statelist is created for that modifier */
	virtual void addStateModifier(CStateModifier *modifier);
/// remove state modifier from the list, the coresponding state list is deleted
	virtual void removeStateModifier(CStateModifier *modifier);

	void loadASCII(FILE *stream);
	void saveASCII(FILE *stream);
	
	void loadBIN(FILE *stream);
	void saveBIN(FILE *stream);

	int getNumStateCollections();
};

#endif

