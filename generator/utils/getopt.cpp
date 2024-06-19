#include "getopt.h"

#include <string.h>
#include <stdio.h>

int     opterr = 1,         
        optind = 1,             
        optopt,                 
        optreset;           
char    *optarg;

#define BADCH   (int)'?'
#define BADARG  (int)':'
#define EMSG    ""

int getopt(int nargc, char * const nargv[], const char *ostr) {
    static char *place = EMSG;              
    const char *oli;                   

    if (optreset || !*place) {              
        optreset = 0;
        if (optind >= nargc || *(place = nargv[optind]) != '-') {
            place = EMSG;
            return (-1);
        }
        if (place[1] && *++place == '-') {     
            ++optind;
            place = EMSG;
            return (-1);
        }
    }                                    
    if ((optopt = (int)*place++) == (int)':' ||
        !(oli = strchr(ostr, optopt))) {
        if (optopt == (int)'-')
            return (-1);
        if (!*place)
            ++optind;
        if (opterr && *ostr != ':')
            (void)printf("illegal option -- %c\n", optopt);
        return (BADCH);
    }
    if (*++oli != ':') {                    
        optarg = NULL;
        if (!*place)
        ++optind;
    } else {                                  
        if (*place) optarg = place;
        else if (nargc <= ++optind) {   
            place = EMSG;
            if (*ostr == ':')
                return (BADARG);
            if (opterr)
                (void)printf("option requires an argument -- %c\n", optopt);
            return (BADCH);
        }
        else optarg = nargv[optind];
        place = EMSG;
         ++optind;
    }
    return (optopt);                        
}