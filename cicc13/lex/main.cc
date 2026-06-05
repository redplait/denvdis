#include <stdio.h>
#include "myscanner.h"

extern "C" int yylex(void);

int main(void)
{
    int ntoken,vtoken;

    ntoken=yylex();
    while(ntoken)
    {
	printf("%d\n",ntoken);
	ntoken=yylex();
    }
    return 0;
}