#include "RBFBasedQFunction.h"
#include "RBFQETraces.h"
	


CAbstractQETraces* RBFBasedQFunctionBinary::getStandardETraces()
{
    return new RBFQETraces(this);
}
