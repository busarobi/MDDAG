#include "cnearestneighbor.h"

#include <limits>
#include <assert.h>
#include <math.h>

CKDRectangle::CKDRectangle(int numDim)
{
	minValues = new ColumnVector(numDim);
	maxValues = new ColumnVector(numDim);
	
	(*minValues) = -numeric_limits<double>::max();
	(*maxValues) = numeric_limits<double>::max();
}

CKDRectangle::CKDRectangle(CKDRectangle &rectangle)
{
	minValues = new ColumnVector(*rectangle.getMinVector());
	maxValues = new ColumnVector(*rectangle.getMaxVector());
}

CKDRectangle::~CKDRectangle()
{
	delete minValues;
	delete maxValues;
}

ColumnVector *CKDRectangle::getMinVector()
{
	return minValues;
}

ColumnVector *CKDRectangle::getMaxVector()
{
	return maxValues;
}


bool CKDRectangle::intersects(CKDRectangle *rectangle)
{
	for (int i = 0; i < minValues->nrows(); i ++)
	{
		if ((rectangle->getMaxValue(i) < minValues->element(i)) || ( rectangle->getMinValue(i) > maxValues->element(i)))
		{
/*			printf("No Intersection %d:", i);
			cout << getMinVector()->t() << "; " << getMaxVector()->t() << endl;

			cout << rectangle->getMinVector()->t() << "; " << rectangle->getMaxVector()->t() << endl;
*/

			return false;
		} 
	}
	return true;
}

void CKDRectangle::setMaxValue(int dim, double value)
{
	maxValues->element(dim)  = value;
}

void CKDRectangle::setMinValue(int dim, double value)
{
	minValues->element(dim)  = value;
}


double CKDRectangle::getMinValue(int dim)
{
	return minValues->element(dim);
}

double CKDRectangle::getMaxValue(int dim)
{
	return maxValues->element(dim);
}

double CKDRectangle::getDistanceToPoint(ColumnVector *point)
{
	double distance = 0;
	for (int i = 0; i < point->nrows(); i ++)
	{
		if (point->element(i) < minValues->element(i))
		{
			distance += pow(point->element(i) - minValues->element(i), 2.0);
//			printf("min : %d %f %f %f, ", i, point->element(i) - minValues->element(i), point->element(i), minValues->element(i));
		}
		else
		{
			if (point->element(i) > maxValues->element(i))
			{
				distance += pow(point->element(i) - maxValues->element(i), 2.0);
			
//				printf("max : %d %f %f %f, ", i, point->element(i) - minValues->element(i), point->element(i), minValues->element(i));
			}
		}
	}
	return sqrt(distance);
}

void CKNearestNeighbors::addDataElements(ColumnVector *point, CLeaf<DataSubset *> *leaf, CKDRectangle *)
{
	DataSubset *subset = leaf->getTreeData();
	DataSubset::iterator it = subset->begin();
	for (; it != subset->end(); it ++)
	{
		*buffVector = *point - (*(*inputSet)[*it]);
		
		addAndSortDataElements(*it, buffVector->norm_Frobenius());
	}
}
		
CKNearestNeighbors::CKNearestNeighbors(CTree<DataSubset *> *tree, CDataSet *l_inputSet, int K) : CKNearestNeighborsTreeData<int, DataSubset *>(tree, K)
{
	inputSet = l_inputSet;	
	buffVector = new ColumnVector(inputSet->getNumDimensions());
}

CKNearestNeighbors::~CKNearestNeighbors()
{
	delete buffVector;
}

CRangeSearch::CRangeSearch(CTree<DataSubset *> *l_tree, CDataSet *l_inputSet)
{
	tree = l_tree;
	inputSet = l_inputSet;
	
	elementList = new DataSubset;

}

CRangeSearch::~CRangeSearch()
{
	delete elementList;
}

void CRangeSearch::getSamplesInRangeElements(CKDRectangle *range, CTreeElement<DataSubset *> *element, CKDRectangle *rectangle)
{
	if (element->isLeaf())
	{
		CLeaf<DataSubset *> *leaf = dynamic_cast<CLeaf<DataSubset *> *>(element);
		addDataElements(range, leaf, rectangle);
	}
	else
	{
		CNode<DataSubset *> *node = dynamic_cast<CNode<DataSubset *> *>(element);	
		C1DSplittingCondition *split = dynamic_cast<C1DSplittingCondition *>(node->getSplittingCondition());

		double minValue = rectangle->getMinValue(split->getDimension());
		double maxValue = rectangle->getMaxValue(split->getDimension());

		if (split->getTreshold() < minValue || split->getTreshold() > maxValue)
		{
			printf("Split Not in Rectangle: %f: %f %f\n", split->getTreshold(), minValue, maxValue);
			assert(false);
		}

		// set value for left branch
		rectangle->setMaxValue(split->getDimension(), split->getTreshold());

		bool leftIntersect = rectangle->intersects(range);

		if (leftIntersect)
		{
			getSamplesInRangeElements(range, node->getLeftElement(), rectangle);
		}

		rectangle->setMaxValue(split->getDimension(), maxValue);
		rectangle->setMinValue(split->getDimension(), split->getTreshold());

		bool rightIntersect = rectangle->intersects(range);

		if (rightIntersect)
		{
			getSamplesInRangeElements(range, node->getRightElement(), rectangle);	
		}
		
		//printf("LeftDist: %f, RightDist: %f...", leftDist, rightDist);	
		/*
		for (int i = 0; i < K; i ++)
		{
			if (distList[i] < numeric_limits<double>::max())
			{
				printf("%f ", distList[i]);
			}
		}
		printf("\n");*/

		

		rectangle->setMaxValue(split->getDimension(), maxValue);
		rectangle->setMinValue(split->getDimension(), minValue); 
	}
}
	
void CRangeSearch::addAndSortDataElements(int data)
{
	
	elementList->insert(data);

	
	/*printf("Adding Element with distance %f\n", distance);
	for (int i = 0; i < K; i ++)
	{
		if (distList[i] < numeric_limits<double>::max())
		{
			printf("%f ", distList[i]);
		}
	}
	printf("\n");*/
}


void CRangeSearch::getSamplesInRange(CKDRectangle *range, DataSubset *l_elementList)
{
	elementList->clear();
	
	
	CKDRectangle rectangle(tree->getNumDimensions());
	
	CKDRectangle pre_range(*range);
	
	if (tree->getPreprocessor())
	{
		tree->getPreprocessor()->preprocessInput(pre_range.getMinVector(), pre_range.getMinVector());
		tree->getPreprocessor()->preprocessInput(pre_range.getMaxVector(), pre_range.getMaxVector());
	}

/*	printf("Starting Range search for area:\n");
	cout << pre_range.getMinVector()->t() << endl;	
	cout << pre_range.getMaxVector()->t() << endl;		
*/
	getSamplesInRangeElements(&pre_range, tree->getRoot(), &rectangle);
	
	l_elementList->clear();
	
	DataSubset::iterator it = elementList->begin();

//	printf("Points in range %d: \n", elementList->size());
	for (; it != elementList->end(); it ++)
	{
		l_elementList->insert(*it);
//		cout << (*inputSet)[*it]->t() << endl;
	}

/*	CDataSet::iterator it2  = inputSet->begin();

	int numElements = 0;
	for (;it2 != inputSet->end(); it2 ++)
	{
		if (pre_range.getDistanceToPoint(*it2) == 0)
		{
			numElements ++;
		}
	}
	printf("Real Points in Range: %d, %d\n", numElements, elementList->size());*/
}

void CRangeSearch::addDataElements(CKDRectangle *range, CLeaf<DataSubset *> *leaf, CKDRectangle *leafRectangle)
{
	DataSubset *subset = leaf->getTreeData();
	DataSubset::iterator it = subset->begin();

	for (; it != subset->end(); it ++)
	{
		double distance = range->getDistanceToPoint((*inputSet)[*it]);
		
		if (distance == 0)
		{
			addAndSortDataElements(*it);
		}
	}
}

