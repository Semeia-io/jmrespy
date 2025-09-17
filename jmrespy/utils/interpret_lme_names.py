"""
Permite to interpret names from lme_object (usefull when we have interactions or categorical variables for exemple)
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np
import pandas as pd

def Interpret_lme_names(lme_datas, lme_names):
    
    #List which will contain all vectors of out output dataframe
    lst_out = []
    for col in lme_names:

        #First, we carry the multiplication (for associations)
        if ':' in col:
            lst_multiple = col.split(':')
            if len(lst_multiple) > 2:
                raise ValueError("Error in {}, can't multiply more than two columns in same time".format(col))
            else:
                #Left argument of multiplication
                if '[T.' in lst_multiple[0] and lst_multiple[0][-1] == ']':
                    multpile_left_1 = lst_multiple[0].split('[T.')[0]
                    multiple_left_2 = lst_multiple[0].split('[T.')[1][:-1]
                    vec1 = eval("(lme_datas['{}']=='{}').astype(int)".format(multpile_left_1, multiple_left_2))
                else:
                    multpile_left = lst_multiple[0]
                    vec1 = eval("lme_datas['{}']".format(multpile_left))
                #Right argument of multiplication
                if '[T.' in lst_multiple[1] and lst_multiple[1][-1] == ']':
                    mutliple_right = lst_multiple[1].split('[T.')[:-1]
                    multpile_right_1 = lst_multiple[1].split('[T.')[0]
                    multiple_right_2 = lst_multiple[1].split('[T.')[1][:-1]
                    vec2 = eval("(lme_datas['{}']=='{}').astype(int)".format(multpile_right_1, multiple_right_2))
                else:
                    multiple_right = lst_multiple[1]
                    vec2 = eval("lme_datas['{}']".format(multiple_right))
                lst_out.append(list(vec1 * vec2))
        
        #If we haven't multiplication, then we carry categorical variable
        elif '[T.' in  col  and col[-1] == ']':
            arg_1 = col.split('[T.')[0]
            arg_2 = col.split('[T.')[1][:-1]
            vec = eval("(lme_datas['{}']=='{}').astype(int)".format(arg_1, arg_2))
            lst_out.append(list(vec))
        
        #And finally, we carry the classical simplest expected variables in lme models
        else:
            vec = eval("lme_datas['{}']".format(col))
            lst_out.append(list(vec))

    #Converting our list in pandas dataframe and return it
    out = pd.DataFrame(np.array(lst_out).T, columns=lme_names)
    return(out)