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

#ifndef C_FEATUREFUNCTION_H
#define C_FEATUREFUNCTION_H

#include <stdio.h>
#include <list>
#include <map>

/// Class for storing a single feature (feature Index and feature Faktor
/** Used for creating a sparse array of features in comibintation of CFeatureList. 
*/
class CFeature
{
public:
	CFeature();
	CFeature(unsigned int Index, double factor);

	~CFeature();


	unsigned int featureIndex;
	double factor;
};

/// Class for storing features as sparse array
/** 
The class maintains represents a list of features and is often used for gradient calculatation. In a feature list, not all features have to be included, the features which are not in the list have automatically the value 0.0. Every feature in the list is stored as a CFeature object, so the index and the feature factor are stored. The CFeatureList class also contains methods for directly setting (set), updating (adding a factor, update) or getting (getFeatureFactor) feature factors, without the need of CFeature objects. CFeatureList implements all functionality of the std::list class, so iterators can be used in the same way.
The class provides its own resource management, so it always uses its own CFeature objects, so you don't have to care about where the CFeature objects come from or where they will go (cleaning up). If the feature list gets too small, it automatically allocates new resources. If the list gets smaller again, it always holds the allocated resources (even if you call clear) because its likely that the resources will be needed again. The allocated CFeature resources get deleted only when the feature list itself is deleted or clearAndDelete is called. For performance reasons the feature list also contains a map of CFeature object, with its feature index as search index. So the access to a particular feature is very quick. All features that are in the list are always in the map too.  
You have also the possibility to add another feature list to the current list (feature factors are added, non existant features in one list count as feature factor 0, or to multiply 2 feature lists (needed for gradient multiplication). The multiplied feature list naturally contains only the features which are in both components of the product.
Feature lists can also be sorted by the feature factor. This is done for example in the case of etraces. If you need a sorted feature list, set the isSortet flag of the constructor to true. The first feature in the list will then always be the one with the highest factor.
*/

class CFeatureList  : protected std::list<CFeature *>
{
protected:
	std::list<CFeature *> *freeResources;
	std::list<CFeature *> *allResources;
	std::map<int, CFeature *> *featureMap;
	bool isSorted;
	bool sortAbs;

	/// Internal function, used for resource management
	/** 
	Returns an unused or a new CFeature object.
	*/
	CFeature *getFreeFeatureResource();

	/// insert the given feature in the correct position.
	void sortFeature(CFeature *feature);

	
public:
	typedef CFeatureList::iterator iterator;
	typedef CFeatureList::reverse_iterator reverse_iterator;

/// Create a new feature list
	/** 
	The parameter initMemSize specifies the number of CFeatures object that are allocated in the beginning. The second flag specifies wether the list is sorted.
	*/
	CFeatureList(int initMemSize = 0, bool isSorted = false, bool sortAbs = false);
	~CFeatureList();

	///Returns iterator pointing on the given feature (if its in the list, other wise end will be returned)
	CFeatureList::iterator getFeaturePos(unsigned int feature);

	/// Add a CFeature object to the list
	/** 
	The value of the feature is added to the feature in the list or, if the feature isn't already in the list, a new feature is added to the list.
	Even though a CFeature object is given, the feature list uses its own CFeature objects. 
	*/
	void add(CFeature *feature);
	/// Add all features of the given list to the current feature list. Thre feature factors of the given list are multiplied by "factor".
	void add(CFeatureList *featureList, double factor = 1.0);
	/// Set the given feature to the given factor
	/**
	If the feature is not already in the list a new CFeature object is requested (by the function getFreeFeatureResource.
	*/
	void set(int feature, double factor);

	/// Multiply all feature factors with the given factor
	void multFactor(double factor);

	/// Multiply all feature of the given list with the features of the current list.
	/** 
	This calculation is like the dot product, features that are not in a list have the feature factors 0.0.
	*/
	double multFeatureList(CFeatureList *featureList);
	
	/// Add to all features the given index offset.
	void addIndexOffset(int Offset);

	/// Add the given feature factor to the given feature
	/** 
	If the feature is not already in the list a new CFeature object is requested (by the function getFreeFeatureResource) and the feature factor is set according to factor.

	*/
	void update(int feature, double factor);

	/// Returns the feature factor of the given feature
	double getFeatureFactor(int featureIndex);
	/// Returns the whole CFeature object with the given factor
	/** 
	If the specified feature isn't in the list, NULL is returned. Don't change any values or delete that CFeature object !!!
	*/
	CFeature* getFeature(int featureIndex);

	/// removes the given feature from the list
	void remove(CFeature *feature);
	void remove(int feature);

	/// clears the feature list
	/** 
	The allocated CFeature objects remain in memory, they are just removed from the list and put in the freeResources list.
	*/
	void clear();
	/// clears the feature list and deletes all CFeature objects
	void clearAndDelete();

	/// save the feature list to a ascii stream
	void saveASCII(FILE *stream);
	/// load the feature list to a ascii stream
	void loadASCII(FILE *stream);

	/// normalizes the feature list (sum of all factors = 1)
	void normalize();

	/// returns the euklidean length of the feature space
	double getLength();


	CFeatureList::iterator begin();
	CFeatureList::iterator end();
	CFeatureList::reverse_iterator rbegin();
	CFeatureList::reverse_iterator rend();

	int size() {return std::list<CFeature *>::size();}
};


/// The feature function for storing features in an double array.
/** This class is base class of all V-Functions which use features (used by linear approximators) and discrete states. A feature function is a table storing the values of every feature. The class provides direct access to the feature values through the functions setFeature, updateFeature and getFeature. It also provides functions for working with feature lists (setFeatureList, updateFeatureList, getFeatureList). When working with feature lists, not a single feature value, but all feature values of the features in the list get accessed, but each access is "multiplied" by the features activation factors. So, for example if you want to update the features of a feature list by the factor 5.0, and the feature list contains two features, feature nr. 80 and feature nr. 85, each having the same activation factor (often also called feature factor) of 0.5, than the update for both features would be 2.5. The same concept is true for setFeatureList and getFeatureList.
*/
class CFeatureFunction 
{
protected:
	/// number of features
	unsigned int numFeatures;
	/// Feature double array
	double *features;

	bool externFeatures;

public:
/// creates a feature function with numFeatures features.
	CFeatureFunction(unsigned int numFeatures);
	CFeatureFunction(unsigned int numFeatures, double *features);

	virtual ~CFeatureFunction();

/// Initializes the features with random values
/**
The random values are sampled from an uniform distribution between min and max.
*/
	void randomInit(double min = -1.0, double max = 1.0);

	void init(double value);

/// Sets the feature to the specified value
/** the value gets multiplied by the feature factor of update object.
*/
	void setFeature(CFeature *update, double value);
/// Sets the feature to the specified value
	void setFeature(unsigned int featureIndex, double value);
/// Sets all features of the list to the specified values
/** calls setFeature(CFeature *update, double value), so the value gets
multiplied by the feature factor for each feature.
*/
	void setFeatureList(CFeatureList *updateList, double value);

/// Adds the difference to the specified feature.
	void updateFeature(int feature, double difference);
/// Adds the difference to the specified feature.
/**
The difference is multiplied by the feature factor of the update object before updating.*/
    void updateFeature(CFeature *update, double difference);
/// Adds the difference to all features in the feature list.
/**
Calls updateFeature(CFeature *update, double difference), so the difference is multiplied by the feature factor before updating.*/
	void updateFeatureList(CFeatureList *updateList, double value);

/// Returns the value of the feature
	virtual double getFeature(unsigned int featureIndex);

/// Returns the summed values of the features in the list.
/** Each value of a feature gets multplied by the feature factor and then summed up.
*/
	virtual double getFeatureList(CFeatureList *featureList);

	virtual void saveFeatures(FILE *stream);
        virtual void loadFeatures(FILE *stream);
 	virtual void printFeatures();

	virtual unsigned int getNumFeatures();

	void postProcessWeights(double mean, double std);
};


#endif

