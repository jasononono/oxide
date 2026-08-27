// this pattern is pasted underneath functions that need  specialization


#ifdef SPEC2D
#define MACRO TEMPLATE2D
#else
#define MACRO TEMPLATE
#endif

MACRO(int32)
MACRO(float32)

#ifndef NO_UNDEF
#undef TEMPLATE
#endif
#undef MACRO