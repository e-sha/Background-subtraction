#include <BaseSubtractorTester.h>

BaseSubtractorTester::BaseSubtractorTester()
	: m_subtractor(nullptr)
{
}

void BaseSubtractorTester::SetSubtractor(BaseSubtractor *in_subtractor)
{
	m_subtractor = in_subtractor;
}
