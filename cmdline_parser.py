

##################################
## DEPENDENCIES
##################################
import os
import sys
import glob

##################################
## FLAGS
##################################
DEBUG = False

##################################
## FUNCTIONS
##################################
def parse(cmdline, min_args, PARAMS, FLAGS, FILES, DEBUG = DEBUG):
    EXIT_CODE = -1 ## -1 = failed, 1 = passed
    if DEBUG:
        print("Command line arguments = ")
        print(" ", cmdline)

    ## check if help flag was called or we have a minimum number of arguments to evaluate
    if len(cmdline) - 1 < min_args or check_for_help_flag(cmdline):
        print(" Not enough arguments, or help flag called")
        return PARAMS, EXIT_CODE

    ## check if a flag appears twice
    all_flags = []
    for cmd in cmdline:
        if cmd in FLAGS:
            all_flags.append(cmd)
    if len(all_flags) != len(set(all_flags)):
        print(" ERROR :: A flag appears twice in input!")
        return PARAMS, EXIT_CODE

    ## figure out file set up from cmd line and if batch mode is being used instead of explicit entry
    if check_batchmode(FILES)[0]:
        ## if we can batch, then we need to check with priority for '@' token against the expected extensions
        extensions = check_batchmode(FILES)[1]
        if not isinstance(extensions, list):
            extensions = make_list(extensions)
        for allowed_extension in extensions:
            for cmd in cmdline:
                if DEBUG:
                    print("CHECK BATCH MODE, cmd = %s; allowed_extension = %s " % (cmd, allowed_extension))
                if len(cmd) > 4: ## lowest cmd size is 5, e.g.: '@.ext'
                    if cmd[-5:] == '@' + allowed_extension:
                        PARAMS['BATCH_MODE'] = True
                        print(" ... batch mode enabled with: ", cmd)
                        break

    ## if batch mode is active, we dont need to parse names for the files, they will be created by glob matching and string manipulation at runtime
    if DEBUG:
        print("==========================================")
        print(" PARSE FILES FROM CMD LINE")
        print("==========================================")
    if not 'BATCH_MODE' in PARAMS or not PARAMS['BATCH_MODE']:
        ## if batch mode is off, we need to find explicit entries for each expected file
        for file in FILES:
            cmdline_index = FILES[file][0]
            extensions = FILES[file][1]
            ## sanity check the requested index for the expected file can actually exist 
            if cmdline_index >= len(cmdline):
                print(" !! ERROR :: Could not find input file (allowed extensions = %s) expected at argument $%s" % (extensions, cmdline_index))
                return PARAMS, EXIT_CODE

            if cmdline_index > 0:
                ## inflexible indexing, check extension at specific position on cmd line
                MATCH = False
                for allowed_extension in extensions:
                    if len(cmdline[cmdline_index]) > len(allowed_extension) - 1: ## lowest cmd size is 4, e.g.: '.ext'
                        if cmdline[cmdline_index][-len(allowed_extension):] == allowed_extension:
                            PARAMS[file] = cmdline[cmdline_index]
                            if DEBUG:
                                print(" ... assigned '%s' = %s " % (file, cmdline[cmdline_index]) )
                            MATCH = True
                if not MATCH:
                    print(" ERROR :: File at $%s (%s) not expected format: %s" % (cmdline_index, cmdline[cmdline_index], extensions))
                    return PARAMS, EXIT_CODE
            else:
                ## flexible indexing, find file based on extension alone
                for allowed_extension in extensions:
                    for cmd in cmdline:
                        if len(cmd) > 3: ## lowest cmd size is 4, e.g.: '.ext'
                            if cmd[-(len(allowed_extension)):] == allowed_extension:
                                PARAMS[file] = cmd
                                if DEBUG:
                                    print(" ... assigned '%s' =  %s" % (file, cmd))

            # print("file key, file value: ", file, FILES[file])


    ## iterate over all arguments and parse all flags
    for i in range(len(cmdline)):
        cmd = cmdline[i]
        if cmd in FLAGS:
            PARAMS, FLAG_EXIT_CODE = parse_flag(cmdline, i, PARAMS, FLAGS[cmd], FLAGS, FILES)
            if FLAG_EXIT_CODE < 0:
                print(" ERROR :: Could not correcly parse flag: %s" % cmd)
                return PARAMS, EXIT_CODE
        ## warn the user if a flag-like entry is found, incase they had a typo; proceed as normal otherwise
        elif "--" in cmd:
            print(" WARNING :: unexpected flag not parsed: (%s)" % cmd)


    EXIT_CODE = 1
    return PARAMS, EXIT_CODE

def check_for_help_flag(cmdline):
    for entry in cmdline:
        if entry in ['-h', '-help', '--h', '--help']:
            if DEBUG:
                print(' ... help flag called (%s).' % entry)
            return True
    return False

def check_batchmode(FILES):
    extensions = None
    for fname in FILES:
        can_batch_mode = FILES[fname][2]
        if can_batch_mode:
            extensions = FILES[fname][1]
            return True, extensions
    return False, extensions

def parse_flag(cmdline, index, PARAMS, flag_options, FLAGS, FILES):
    EXIT_CODE = -2
    if DEBUG:
        print("==========================================")
        print(" PARSE FLAG FROM CMD LINE :: %s " % cmdline[index])
        # print("  ... flag_options = ", flag_options)
        print("==========================================")

    ## remap variables and make them into lists if not already
    PARAM_keys = make_list(flag_options[0])
    datatypes = make_list(flag_options[1])
    legal_entries = make_list(flag_options[2])
    input_toggles = make_list(flag_options[3]) ## toggles connected to specific inputs to the flag
    intrinsic_toggle = flag_options[4] ## toggle connected to flag, regardless of inputs
    has_defaults = flag_options[5]

    for i in range(len(PARAM_keys)):
        PARAM_key = PARAM_keys[i]
        ## check if there are even enough entries to parse past the flag itself
        if len(cmdline) > index + 1 + i:
            string_to_parse = cmdline[index + 1 + i]
        else:
            string_to_parse = None
        ## dont let a flag accidentally become parsed as an entry
        if string_to_parse in FLAGS:
            string_to_parse = None
            break
        ## dont let a file accidentally become parsed as an entry
        list_of_files = []
        for file in FILES:
            list_of_files.append(PARAMS[file])
        if string_to_parse in list_of_files:
            string_to_parse = None
            break

        if DEBUG:
            print("    ... Try setting '%s' to: %s" % (PARAM_key, string_to_parse))
        ## 1. Recast the variable, if necessary, to proper type
        input, EXIT_CODE_TRY_CAST = try_cast(string_to_parse, datatypes[i])
        if EXIT_CODE_TRY_CAST < 0:
            ## check if there are default values
            if has_defaults:
                if DEBUG:
                    print("    ... unexpected or no entry given, using defaults: '%s'" % PARAMS[PARAM_key])
                    print("  >> set '%s' = '%s'" % (PARAM_key, PARAMS[PARAM_key]))
            else:
                ## otherwise throw error
                print(" ERROR :: Could not cast '%s' as correct type" % string_to_parse )
                return PARAMS, EXIT_CODE
        else:
            ## if we cast correctly, then proceed with trying to read in the input
            ## 2. Check if the parsed input is in a legal range
            PARAMETER_SET = False ## flag to determine if script should try to load a default value
            ## deal with numbers differently than string
            if not isinstance(input, str):
                ## ints and floats use a range check
                if DEBUG:
                    print("    ... TEST if %s is in range (%s, %s)" % (input, legal_entries[i][0], legal_entries[i][1]))
                if legal_entries[i][0] <= input <= legal_entries[i][1]:
                    PARAMS[PARAM_key] = input
                    PARAMETER_SET = True
                    if DEBUG:
                        print("  >> set '%s' = %s" % (PARAM_key, input))
            else:
                ## check if there are any options at all, if not that any input is valid
                if not len(legal_entries[i]) > 0:
                    PARAMS[PARAM_key] = input
                    PARAMETER_SET = True
                    if DEBUG:
                        print("  >> set '%s' = %s" % (PARAM_key, input))
                else:
                    if DEBUG:
                        print("    ... check if input exists as an option:", legal_entries[i])
                    ## strings check against a library of options
                    if input in legal_entries[i]:
                        PARAMS[PARAM_key] = input
                        PARAMETER_SET = True
                        if DEBUG:
                            print("  >> set '%s' = %s" % (PARAM_key, input))

            if not PARAMETER_SET:
                ## check for if a default parameter is enabled
                if has_defaults:
                    if DEBUG:
                        print("    ... unexpected or no entry given, using defaults: '%s'" % PARAMS[PARAM_key])
                        print("  >> set '%s' = '%s'" % (PARAM_key, PARAMS[PARAM_key]))
                else:
                    print(" ERROR :: Flag not parsed (%s), and does allow default values!" % cmdline[index])
                    return PARAMS, EXIT_CODE
            else:
                ## if defaults are not used, activate any toggles connected to this flag input
                if input_toggles[i]:
                    PARAMS[input_toggles[i][1]] = input_toggles[i][2]
                    if DEBUG:
                        print(" Set toggle '%s' = %s" % (input_toggles[i][1], input_toggles[i][2]))

    ## check if the flag has an intrinsic toggle to set
    if intrinsic_toggle:
        ## set the toggle
        PARAMS[intrinsic_toggle[1]] = intrinsic_toggle[2]
        if DEBUG:
            print("  >> set toggle '%s' = %s " % (intrinsic_toggle[1], intrinsic_toggle[2]))
    return PARAMS, 1

def try_cast(string, type):
    EXIT_CODE = -3
    recast_var = None
    if string == None:
        return recast_var, EXIT_CODE
    ## default input is a string, so no work to do here
    if isinstance(type, str):
        return string, 3
    ## try to recast input as integers/floats
    if isinstance(type, int):
        try:
            recast_var = int(string)
            return recast_var, 3
        except:
            # print(" ERROR :: Could not cast %s as int" % string )
            return recast_var, EXIT_CODE
    if isinstance(type, float):
        try:
            recast_var = float(string)
            return recast_var, 3
        except:
            # print(" ERROR :: Could not cast %s as float" % string )
            return recast_var, EXIT_CODE
    return recast_var, EXIT_CODE

def make_list(input):
    empty_list = []
    if not isinstance(input, list):
        empty_list.append(input)
        return empty_list
    else:
        return input

def print_parameters(PARAMS, cmdline):
    cmdline_string = ""
    for cmd in cmdline:
        cmdline_string += "  " +  cmd
    print("=========================================================================")
    print("  $" + cmdline_string)
    print("=========================================================================")
    for param in PARAMS:
        print("  %s : %s" % ("{:<20}".format(param[:20]), PARAMS[param]))
    print("=========================================================================")
    return

##################################
## TEST PARSER
##################################
if __name__ == "__main__":

    FLAGS = {
     '--var1' : (
    ['length', 'width', 'title'], ## list of expected entries and their corresponding PARAMS key (use len() to find total inputs to look for a given flag)
    [float(), float(), str()], ## list of datatypes expected for each input entry
    [(0,999), (0,999), ()], ## legal range of values that can be taken (empty tuple == no limit to range
    [False, (True, 'XY_GIVEN', True), (True, 'TITLE_GIVEN', True)], ## if any specific entry connects to a toggle PARAM key
    False, ## if the flag itself (without any options given) should connect to a toggle PARAM key
    True ## if the flag has default params (e.g. allow the user to call the flag without any inputs)
     ),
     '--var2' : (
    'option', ## list of expected entries and their corresponding PARAMS key (use len() to find total inputs to look for a given flag)
    str(), ## list of datatypes expected for each input entry
    ('a', 'b', 'c'), ## legal range of values that can be taken (empty tuple == no limit to range
    False, ## if any entry connects to a toggle PARAM key
    (True, 'OPTION_ACTIVE', True), ## calling the flag alone will set this toggle to the tuple entry at index 2
    False ## if the flag has default params (e.g. allow the user to call the flag without any inputs)
     ),
        '--j' : (
 	'threads',
    int(),
    (0,999),
    False,
    (True, 'PARALLEL_PROCESSING', True),
    True
    )
    }

    FILES = {
    'input_file' : (
    -1, ## expected index (-1 indicates no fixed index, check across all input commands; otherwise $1 == 1, $2 == 2 and so on)
    ['.mrc','.ser'], ## allowed extensions
    False ## if entry can launch batch mode using a @ symbol as the input file base name, entries that cannot are technically optional
    ),
    'output_file' : (
    -1,
    ['.jpg','.png','.tif', '.gif'],
    True ## can launch batch mode via: '@' + ['.jpg','.png','.tif', '.gif'], as input (NOTE: only 1 file/param should ever be able to launch batch mode atm
    ),
    }

    PARAMS = { 			  ## set defaults here
    'input_file' 			: str(),
    'output_file' 			: str(),
    'BATCH_MODE' 			: False, ## BATCH_MODE keyword is hardcoded into parser by default, using '@.ext' as identifier
    'XY_GIVEN' 				: False,
    'length' 				: 4,
    'width'  				: 4,
    'TITLE_GIVEN' 			: False,
    'title'  				: 'A rectangle of given shape',
    'OPTION_ACTIVE'         : False,
    'option'                : 'a',
    'PARALLEL_PROCESSING' 	: False,
    'threads'               : 4
    }


    ## test this script with all options:
    # cmdline = ['cmdline_parser.py', '--var2', 'test.ser', '--var1', 'test.png', '--j']
    # cmdline = ['cmdline_parser.py', '--var1', '4', '2.3', 'new title', 'test.ser', '--var2', 'test.png', '--j']
    # cmdline = ['cmdline_parser.py', '--var1', '4', 'test.ser', '--var2', 'wrong_option', 'test.png', '--j', '8']
    # cmdline = ['cmdline_parser.py', '--var1', '@.ser', '--var2', 'test.png', '--j']
    # cmdline = ['cmdline_parser.py', '--var1', 'input.ser', '--var2', '@.png', '--j']
    # cmdline = ['cmdline_parser.py', '--var1', 'input.ser', '--var2', '@.png']
    PARAMS, EXIT_CODE = parse(cmdline, 2, PARAMS, FLAGS, FILES)
    # print("EXIT CODE = %s" % EXIT_CODE)
    print_parameters(PARAMS, cmdline)
