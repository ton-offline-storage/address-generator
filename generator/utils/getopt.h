#pragma once

#ifndef GETOPT_H
#define GETOPT_H
  
extern int opterr;		
extern int optind;		
extern int optopt;		
extern int optreset;
extern char *optarg;

int getopt(int nargc, char * const nargv[], const char *ostr);

#endif