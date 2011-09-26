#include "ctrees.h" 





C1DSplittingCondition::C1DSplittingCondition(int l_dimension, double l_treshold)
{
	dimension = l_dimension;
	treshold = l_treshold;
}

C1DSplittingCondition::~C1DSplittingCondition()
{
	
}
		
bool C1DSplittingCondition::isLeftNode(ColumnVector *input)
{
	//printf("Split : %d, %f -> %f\n", dimension, treshold, input->element(dimension));
	return input->element(dimension) < treshold;
}


int C1DSplittingCondition::getDimension()
{
	return dimension;
}

double C1DSplittingCondition::getTreshold()
{
	return treshold;
}
