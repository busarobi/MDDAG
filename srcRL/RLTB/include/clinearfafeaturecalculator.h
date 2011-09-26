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

#ifndef CLINEARFAFEATURE_H
#define  CLINEARFAFEATURE_H

#include "cstatemodifier.h"

#include <map>

class CState;
class CStateCollection;

/// Combines 2 or more feature states with an "Or" operatation.
/** 
The "or" operator gives you the possibility to use different feature states simultanously that are independent. You can add several feature calculators, all calculated features from the different modifiers can now be active simultanously in the same feature state. It is understood that you have to distuingish between the different features from the different feature states, so, to get a unique feature index again, the sum off all feature state sizes from the previously added feature states is added to the feature index of a particular feature state. So our feature state size is the sum off all feature state sizes, and the number of active features is naturally the sum of all number of active features.\par
The feature operator or is used to combine 2 or more (more or less) indepedent states, for example tilings or RBF networks with different offsets/resolutions.
\par
You always have to init the feature operator with the function initFeatureOperator after ALL feature states have been added and before you add the feature operator to the agent's modifier list.*/
class CFeatureOperatorOr : public CStateMultiModifier, public CFeatureCalculator
{
protected:
	std::map<CStateModifier *, double> *featureFactors;
public:
	///Creates empty operator
	CFeatureOperatorOr();
	virtual ~CFeatureOperatorOr();

/// Calculates the feature state
	/** 
	The "or" operator gives you the possibility to use different feature states simultanously. You can add several feature calculators, all calculated features from the different modifiers can now be active simultanously in the same feature state. It is understood that you have to distuingish between the different features from the different feature states, so, to get a unique feature index again, the sum off all feature state sizes from the previously added feature states is added to the feature index of a particular feature state. So our feature state size is the sum off all feature state sizes, and the number of active features is naturally the sum of all number of active features.
	*/
	virtual void getModifiedState(CStateCollection *state, CState *modifiedState);
	/// Adds a feature state to the modifier
	/** 
	The state modifier HAS to be a feature calculator or a discretizer. After the operator has been initialized, no feature calculators can be added any more.
	*/
	virtual void addStateModifier(CStateModifier *featCalc, double factor = 1.0);

	/// Inits the feature operator
	/** 
You always have to init the feature operator after ALL feature states have been added and before you use the feature operator (e.g. add the feature operator to the agent's modifier list).	
	*/
	virtual void initFeatureOperator();

	/// returns the feature calculator for a given feature index
	CStateModifier *getStateModifier(int feature);

	/// Doesn't work, not used, use numeric derivations instead.
	//void getFeatureDerivationX(int feature, CStateCollection *state, ColumnVector *targetVector);
};

/// Combines 2 or more feature states with an "And" operatation.
/** 
The "and" operator gives you the possibility to use different feature states simulatanously that are dependent. This class works primary like the discrete operator and class, it combines several dependent feature state to one. The feature state size of the operator is the product off al feature state sizes, the number of active states is the product of all numbers of active states. Each active feature in the operator is assigned to different active states from the feature states (there is one active feature in the operator for each possibility of choosing one active feature from each feature calculator). A new feature index is calculated according to the feature indices of the assigned active features, the feature factor is the product of all feature factors.
\par
The feature operator and is used to combine 2 or more depedent states, for example single state RBF networks (CSingleStateRBFFeatureCalculator).
\par
You always have to init the feature operator with the function initFeatureOperator after ALL feature states have been added and before you add the feature operator to the agent's modifier list.*/
class CFeatureOperatorAnd : public CStateMultiModifier, public CFeatureCalculator
{
protected:
public:
	CFeatureOperatorAnd();

/// Calculates the feature state
/**The "and" operator gives you the possibility to use different feature states simulatanously that are dependent. This class works primary like the discrete operator and class, it combines several dependent feature state to one. The feature state size of the operator is the product off al feature state sizes, the number of active states is the product of all numbers of active states. Each active feature in the operator is assigned to different active states from the feature states (there is one active feature in the operator for each possibility of choosing one active feature from each feature calculator). A new feature index is calculated according to the feature indices of the assigned active features, the feature factor is the product of all feature factors.*/
	virtual void getModifiedState(CStateCollection *state, CState *modifiedState);
	
	/// Adds a feature state to the modifier
	/** 
	The state modifier HAS to be a feature calculator or a discretizer. After the operator has been initialized, no feature calculators can be added any more.
	*/
	virtual void addStateModifier(CStateModifier *featCalc);

	/// Inits the feature operator
	/** 
	You always have to init the feature operator after ALL feature states have been added and before you use the feature operator (e.g. add the feature operator to the agent's modifier list).	
	*/
	virtual void initFeatureOperator();

	/// Doesn't work, not used, use numeric derivations instead.
	//void getFeatureDerivationX(int feature, CStateCollection *state, ColumnVector *targetVector);
};


/// Abstract Superclass for all feature calculator that partition the state space with a grid.
/** 
This class lays a grid over a specified sub-space of the model space. How this grid is used (e.g. Tiling or RBF network) is specified by the sub-classes. You can specify which dimensions you want to use (dimensions array), how much partitions you need for each dimension and also an offset for each dimension which is added to the grid centers. The size of the feature space is the product of all numbers of partitions.
\par
The class provides methods for retrieving the position of a feature (getFeaturePosition), retrieving the active feature (feature nearest to the current state) (getActiveFeature), and retrieving the current partitions of all dimensions (getSingleActiveFeature).
\par
This class always works with the normalized continuous state variable values (scaled to the intervall [0, 1], so consider that when choosing your offsets. 
*/
class CGridFeatureCalculator : public CFeatureCalculator
{
protected:
	unsigned int *partitions;
	unsigned int *dimensions;
	double *offsets;
	
	unsigned int numDim;

	unsigned int *dimensionSize;

	double *gridScale;

public:
	
	/** 
	This class lays a grid over a specified sub-space of the model space. How this grid is used (e.g. Tiling or RBF network) is specified by the sub-classes. You can specify which dimensions you want to use, how much partitions you need for each dimension and also an offset for each dimension which is added to the grid centers. The first parameter is the number of dimensions you want to use so its the size of your arrays. The size of the feature space is the product of all numbers of partitions.
	*/
	CGridFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[], unsigned int numActiveFeatures);
	~CGridFeatureCalculator();

	void setGridScale(int dimension, double scale);

	/// Returns the number of dimensions you use
	int unsigned getNumDimensions();

	/// returns the feature index of the given position array (active partitions of the single state variables)
	int unsigned getFeatureIndex(int position[]);

	/// Stores the position of the specified feature in the position vector
	virtual void getFeaturePosition(unsigned int feature, ColumnVector *position);
	
	/// Returns the number of the nearest feature to the current state
	virtual unsigned int getActiveFeature(CState *state);
	
	/// retrievs the current partitions of all dimensions
	virtual void getSingleActiveFeature(CState *state, unsigned int *activeFeature);
	
	/// Interface method for subclasses
	virtual void getModifiedState(CStateCollection *state, CState *featState) = 0;
};


/// Tilings represent a grid layed over the state space
/** 
The grid is specified the same way as it is in the superclass. A tiling always has just one active state (factor = 1.0), which is the feature containing the current model state. So a single tiling is exactly seen a discrete state. Use CFeatureOperatorOr to combine more tiling objects.
*/
class CTilingFeatureCalculator : public CGridFeatureCalculator
{

public:
	CTilingFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[]);
	~CTilingFeatureCalculator();
	
	/// Returns the active feature
	virtual void getModifiedState(CStateCollection *state, CState *featState);
};

/// Super class for all feature calculators that use a grid and where more than one feature can be active.
/** 
If we use CLinearMultiFeatureCalculator more than one feature can be active in the grid. The class mantains an additional array which mantains the number of active features in each direction, so a value of 1 for dimension 0 means that there are 3 active features for dimension 0 (the current feature itself and it's left and right neighbors) . This array has to be set by the subclasses with the function initAreaSize (which has also to be called by the subclasses in the constructor). For each feature that is in the "active" area, the feature factor is calculated by the function getFeatureFactor, which also has to be implemented by the subclasses.

*/
class CLinearMultiFeatureCalculator : public CGridFeatureCalculator
{
protected:
	unsigned int *areaSize;
	unsigned int areaNumPart;

	ColumnVector *activePosition;
	ColumnVector *featurePosition;
	unsigned int *actualPartition;
	unsigned int *singleStateFeatures;

/// Interface function for returning the feature activation factor
	/** 
	As input the method gets the current model state and the position of the feature as vector.
	*/
	virtual double getFeatureFactor(CState *state, ColumnVector *featPos) = 0;
	/// Interface function for setting the areaSize array
	/** 
	Function has to set the number of active features in each direction. o a value of 1 for dimension 0 means that there are 3 active features for dimension 0 (the current feature itself and it's left and right neighbors).
	Has to be called by the subclass's constructor and has to call the calcNumActiveFeatures method.
	*/
	virtual void initAreaSize();

	/// calculates the number of active feature and stores it in areaNumPart
	/** 
	Has to be called by initAreaSize. Uses the areaSize array to calculate the number of active features. The number of active features is 2 * areaSize[i] + 1 for each dimension i.
	*/
	virtual void calcNumActiveFeatures();
public:
/// For details about creating the grid see the super class.
	CLinearMultiFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[], unsigned int numFeatures);
	~CLinearMultiFeatureCalculator();
	
	/// Calculates the feature state with multiple active features.
	/**
	The class mantains an additional array which mantains the number of active features in each direction, so a value of 1 for dimension 0 means that there are 3 active features for dimension 0 (the current feature itself and it's left and right neighbors) . This array has to be set by the subclasses with the function initAreaSize (which has also to be called by the subclasses in the constructor). For each feature that is in the "active" area, the feature factor is calculated by the function getFeatureFactor, which also has to be implemented by the subclasses. For periodic states the active area is mirrored into the state intervall again if it leaves the intervall, for non-periodic states the active area is cut off. At the end all feature factors get normalized (sum = 1.0)*/
	virtual void getModifiedState(CStateCollection *state, CState *featState);
};

/// Represents a normalized RBF network
/** 
This class lays a grid of RBF-Functions over a specified sub-space of the model space.  You can specify which dimensions you want to use (dimensions array), how much partitions you need for each dimension, an offset for each dimension which is added to the RBF centers and the sigmas for each dimension. When talking about grids, the RBF center is always in the center of the partition. The size of the feature space is the product of all numbers of partitions. 
\par
This class always works with the normalized continuous state variable values (scaled to the intervall [0, 1], so consider that when choosing your offsets and sigmas.
\par
The number of active states for each dimension is choosen according the sigma value. All features are active that are within a range of 2 * sigma. So if you want only the neighbors to be active too, use the thumbrule sigma[i] = 1 / (2 * numPartitions[i] ), so you get 3 active features per dimension.
*/
class CRBFFeatureCalculator : public CLinearMultiFeatureCalculator
{
protected:
	double *sigma;
	double sigmaMaxSize;
	
	/// Calculates the feature factor with the RBF function
	virtual double getFeatureFactor(CState *state, ColumnVector *featPos);
	/// set the areaSize array
	/** 
	The number of active states for each dimension is choosen according the sigma value. All features are active that are within a range of 2 * sigma. So if you want only the neighbors to be active too, use the thumbrule sigma[i] = 1 / (2 * numPartitions[i] ), so you get 3 active features per dimension.
	*/
	virtual void initAreaSize();

public:
	CRBFFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[], double sigma[]);
	CRBFFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[], double sigma[], unsigned int areaSize[]);

	~CRBFFeatureCalculator();

	/// Doesn't work, use numeric derivations instead.
	//virtual void getFeatureDerivationX(int feature, CStateCollection *state, ColumnVector *targetVector);
};

class CLinearInterpolationFeatureCalculator : public CLinearMultiFeatureCalculator
{
	virtual double getFeatureFactor(CState *state, ColumnVector *featPos);
	
	virtual void initAreaSize();

public:
	CLinearInterpolationFeatureCalculator(unsigned int numDim, unsigned int dimensions[], unsigned int partitions[], double offsets[]);
	~CLinearInterpolationFeatureCalculator();

};

/// Superclass for calculating features from a single continuous state variable
/** 
This class allows you, similiar to CSingleStateDiscretizer to calculate your features from a single continuous state variable. Therefore a 1-dimensional grid is layed over the specified continous state variable. How the grid is used is specified by the subclasses (e.g. RBF-Centers). For the current partition of the grid and the neighbors (as much as numActiveFeatures set in the constructor) the feature factor is calculated by the interface function getFeatureFactor. For the current feature and the neighbors calculation periodic state variables get also considered.

At the end the features get normalized (factor sum = 1).
\par
Use CFeatureOperatorOr to combine the feature states comming from different continuous state variables.
*/
class CSingleStateFeatureCalculator : public CFeatureCalculator
{
protected:
	int dimension;
	int numPartitions;
	double *partitions;

	/// Interface function for calculating the feature activation factor
	/** 
	Gets the number of the partition (= featureindex), the value of the continuous state variable and the distance to the feature center to the current state varaible value as input. In this distance periodic states are already considered.
	*/
	virtual double getFeatureFactor(int partition, double contState, double difference) = 0;
public:
	/// Creates the feature calculator
	/** 
	You have to set the dimension of the modelstate you want to calculate your features from, the size of your grid, the grid itself as a double array and how much features should be active. If you set numActiveFeatures for example to 3, the current feature (feature nearest to the current state) and its left and right neighbor are active.
	*/
	CSingleStateFeatureCalculator(int dimension, int numPartitions, double *partitions, int numActiveFeatures);
	virtual ~CSingleStateFeatureCalculator();

	/// Calculates the feature state from one continupus state variable
	/**For the current partition of the grid and the neighbors (as much as numActiveFeatures set in the constructor) the feature factor is calculated by the interface function getFeatureFactor. At the end the features get normalized (factor sum = 1).*/
	virtual void getModifiedState(CStateCollection *state, CState *featState);
};

/// Represents a 1-D RBF-network layed over a specified continous state variable.
/** 
You can specifiy the centers of the RBF-Functions (partitions array) and the sigma of each RBF-Function. The RBF-centers have to be in an ascending order. If you don't specify any sigma, the sigma gets automatically calculated so that the distance between two neighbored features is equal 2 * sigma. 
The rest of the parameters are the same as for the superclass.
*/
class CSingleStateRBFFeatureCalculator : public CSingleStateFeatureCalculator
{
protected:

	
	/// Calculates the feature factor according the RBF formular
	virtual double getFeatureFactor(int partition, double difference, double nextPart);
public:
	CSingleStateRBFFeatureCalculator(int dimension, int numPartitions, double *partitions, int numActiveFeatures);
	

	/// Not working, not used
	//virtual void getFeatureDerivationX(int feature, CStateCollection *state, ColumnVector *targetVector);

};


/// Class for linear interpolation features of a single continuous state variable
/** 
This class represent a linear interpolator for a single continuous state variable. You can specify the centers of the features and which dimension of the modelstate you want to use. There are always 2 features active, the nearest feature to the left and to the right. The feature factors are calculated per linear interpolation, so if dist is the distance of 2 neighbored features and dist_left is the distance to the left feature from the current state x, the feature factor of the left feature is 1 - dist_left / dist and the factor of the right feature is naturally dist_left/dist. 
*/

class CSingleStateLinearInterpolationFeatureCalculator : public CSingleStateFeatureCalculator
{
protected:
	virtual double getFeatureFactor(int partition, double difference, double nextPart);
public:
	CSingleStateLinearInterpolationFeatureCalculator(int dimension, int numPartitions, double *partitions);
	~CSingleStateLinearInterpolationFeatureCalculator();
};

class CFeatureStateNNInput : public CStateModifier
{
protected:
	CFeatureCalculator *featureStateCalc;
	CState *featureState;
public:
	CFeatureStateNNInput(CFeatureCalculator *featureStateCalc);
	~CFeatureStateNNInput();


	virtual void getModifiedState(CStateCollection *state, CState *featState);
};

#endif



