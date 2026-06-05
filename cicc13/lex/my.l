%{
#include "myscanner.h"
%}

%option noyywrap

/* Definitions Section */
DIGIT   [0-9]
HEX_DIG [0-9A-Fa-f]
HEX_NUM 0[xX]{HEX_DIG}+

%%

{HEX_NUM}	return NUM;
[a-zA-Z_][a-zA-Z0-9_$]*		return ID;
".cell"		return CELL;
".eq"		return EQ;
".ne"		return NE;
".async"	return ASYNC;
".col"		return COL;
".row"		return ROW;
%%