/*
 *  ArrayBasedQFunctionBinary.cpp
 *  RLTools
 *
 *  Created by Robert Busa-Fekete on 10/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "ArrayBasedQETraces.h"

template< typename T >
CAbstractQETraces* ArrayBasedQFunctionBinary<T>::getStandardETraces()
{
	return new ArrayBasedQETraces<T>(this);
}

template CAbstractQETraces* ArrayBasedQFunctionBinary<RBFArray<RBF> >::getStandardETraces();
template CAbstractQETraces* ArrayBasedQFunctionBinary<RBFArray<RBFLogScaled> >::getStandardETraces();