# Supported SUBMODES
PN_SUBMODES = {'PrimeFullWindow'         : True,
               'PrimeFullWindowExtended' : True,
               'PrimeLargeWindow'        : True,
               'PrimeSmallWindow'        : False,
               'FastTiming'              : False,
               'FastBurst'               : False,
               'ModifiedTiming'          : False,
               'Large Offset'            : False}

MOS_SUBMODES = {'PrimeFullWindow'  : True,
                'PrimePartialRFS'  : True,
                'PrimePartialW2'   : True,
                'PrimePartialW3'   : True,
                'PrimePartialW4'   : True,
                'PrimePartialW5'   : True,
                'PrimePartialW6'   : True,
                'FastUncompressed' : False,
                'FastCompressed'   : False}

ALL_SUBMODES = PN_SUBMODES.copy()
ALL_SUBMODES.update(MOS_SUBMODES)